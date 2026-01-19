import numpy as np
import torch
import torch.optim as optim

from networks import CategoricalPolicy, ValueFn, rollout_episode, discounted_returns
import gymnasium as gym

def train_reinforce_cartpole(
    total_episodes=600,
    gamma=0.99,
    lr_pi=3e-4,
    lr_v=1e-3,
    hidden=(64,64),
    device="cpu",
    seed=0,
):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    pi = CategoricalPolicy(obs_dim, act_dim, hidden).to(device)
    v  = ValueFn(obs_dim, hidden).to(device)

    opt_pi = optim.Adam(pi.parameters(), lr=lr_pi)
    opt_v  = optim.Adam(v.parameters(), lr=lr_v)

    for ep in range(1, total_episodes + 1):
        obs, act, _, rews = rollout_episode(env, pi, device=device)
        G = discounted_returns(rews, gamma)  # (T,)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        act_t = torch.tensor(act, dtype=torch.int64, device=device)
        G_t   = torch.tensor(G, dtype=torch.float32, device=device)

        # baseline
        v_pred = v(obs_t)
        adv = (G_t - v_pred).detach()

        # policy loss (maximize => minimize negative)
        dist = pi.dist(obs_t)
        logp = dist.log_prob(act_t)
        loss_pi = -(logp * adv).mean()

        # value loss
        loss_v = ((v_pred - G_t) ** 2).mean()

        opt_pi.zero_grad()
        loss_pi.backward()
        opt_pi.step()

        opt_v.zero_grad()
        loss_v.backward()
        opt_v.step()

        if ep % 20 == 0:
            ep_return = rews.sum()
            print(f"[REINFORCE] ep={ep:4d}  return={ep_return:7.1f}")

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
    pi, v = train_reinforce_cartpole()
    watch_policy_human(pi, episodes=3)