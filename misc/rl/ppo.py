import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import CategoricalPolicy, ValueFn, rollout_episode
import gymnasium as gym

def compute_gae(rews, vals, dones, gamma=0.99, lam=0.95):
    # rews, vals are arrays of length T; vals includes V(s_t); dones indicates episode boundary
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = vals[t+1]
        delta = rews[t] + gamma * next_value * next_nonterminal - vals[t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        adv[t] = lastgaelam
    return adv

def collect_trajectories(env, pi, v, batch_steps, device="cpu", gamma=0.99, lam=0.95):
    obs_buf, act_buf, logp_buf, ret_buf, adv_buf, val_buf = [], [], [], [], [], []
    steps = 0
    while steps < batch_steps:
        obs, _ = env.reset()
        done = False
        ep_obs, ep_act, ep_logp, ep_rew, ep_done = [], [], [], [], []
        while not done and steps < batch_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            d = pi.dist(obs_t)
            a = d.sample()
            logp = d.log_prob(a)
            v_s = v(obs_t).item()

            nxt, r, term, trunc, _ = env.step(a.item())
            done = term or trunc

            ep_obs.append(obs)
            ep_act.append(a.item())
            ep_logp.append(logp.item())
            ep_rew.append(r)
            ep_done.append(float(done))

            obs = nxt
            steps += 1

        # bootstrap value
        last_v = 0.0 if done else v(torch.tensor(obs, dtype=torch.float32, device=device)).item()

        # values for GAE need V(s_t) and V(s_{t+1})
        vals = []
        for o in ep_obs:
            vals.append(v(torch.tensor(o, dtype=torch.float32, device=device)).item())
        vals = np.array(vals + [last_v], dtype=np.float32)

        rews = np.array(ep_rew, dtype=np.float32)
        dones = np.array(ep_done, dtype=np.float32)

        adv = compute_gae(rews, vals, dones, gamma=gamma, lam=lam)
        ret = adv + vals[:-1]

        obs_buf.append(np.array(ep_obs, dtype=np.float32))
        act_buf.append(np.array(ep_act, dtype=np.int64))
        logp_buf.append(np.array(ep_logp, dtype=np.float32))
        adv_buf.append(adv)
        ret_buf.append(ret)
        val_buf.append(vals[:-1])

    obs_b = np.concatenate(obs_buf)
    act_b = np.concatenate(act_buf)
    logp_b = np.concatenate(logp_buf)
    adv_b = np.concatenate(adv_buf)
    ret_b = np.concatenate(ret_buf)

    # normalize advantages
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

    return obs_b, act_b, logp_b, adv_b, ret_b

def train_ppo_cartpole(
    total_updates=200,
    steps_per_update=2048,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    lr=3e-4,
    train_pi_iters=10,
    train_v_iters=10,
    minibatch_size=256,
    vf_coef=0.5,
    ent_coef=0.0,
    max_grad_norm=0.5,
    device="cpu",
    seed=0,
):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    pi = CategoricalPolicy(obs_dim, act_dim).to(device)
    v  = ValueFn(obs_dim).to(device)

    opt = optim.Adam(list(pi.parameters()) + list(v.parameters()), lr=lr)

    for upd in range(1, total_updates + 1):
        obs_b, act_b, logp_old_b, adv_b, ret_b = collect_trajectories(
            env, pi, v, steps_per_update, device=device, gamma=gamma, lam=lam
        )

        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_b, dtype=torch.int64, device=device)
        logp_old_t = torch.tensor(logp_old_b, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_b, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_b, dtype=torch.float32, device=device)

        N = obs_t.shape[0]
        idxs = np.arange(N)

        for _ in range(max(train_pi_iters, train_v_iters)):
            np.random.shuffle(idxs)
            for start in range(0, N, minibatch_size):
                mb = idxs[start:start+minibatch_size]
                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_logp_old = logp_old_t[mb]
                mb_adv = adv_t[mb]
                mb_ret = ret_t[mb]

                dist = pi.dist(mb_obs)
                logp = dist.log_prob(mb_act)
                ratio = torch.exp(logp - mb_logp_old)

                # PPO clipped objective (maximize -> negative for minimization)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                loss_pi = -(torch.min(unclipped, clipped)).mean()

                # value loss
                v_pred = v(mb_obs)
                loss_v = ((v_pred - mb_ret) ** 2).mean()

                # entropy bonus (optional)
                ent = dist.entropy().mean()
                loss = loss_pi + vf_coef * loss_v - ent_coef * ent

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(pi.parameters()) + list(v.parameters()), max_grad_norm)
                opt.step()

        # quick evaluation: average of 5 episodes (cheap for CartPole)
        if upd % 10 == 0:
            returns = []
            for _ in range(5):
                _, _, _, rews = rollout_episode(env, pi, device=device)
                returns.append(float(rews.sum()))
            print(f"[PPO] upd={upd:4d}  avg_return_5ep={np.mean(returns):7.1f}")

    env.close()
    return pi, v

def watch_policy_human(policy, episodes=3, device="cpu", seed=0):
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset(seed=seed)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                a = policy.dist(obs_t).probs.argmax().item()  # greedy action for viewing
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_return += r

        print(f"Episode {ep+1}: return={ep_return}")

    env.close()

if __name__ == "__main__":
    pi, v = train_ppo_cartpole()
    watch_policy_human(pi)