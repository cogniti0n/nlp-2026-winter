import numpy as np
import torch
import torch.optim as optim

from networks import CategoricalPolicy, ValueFn, rollout_episode
import gymnasium as gym

def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i+n].view_as(p))
        i += n

def flat_grad(grads):
    return torch.cat([g.contiguous().view(-1) for g in grads])

def conjugate_gradient(Avp_fn, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(iters):
        Avp = Avp_fn(p)
        alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Avp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x

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

def train_trpo_cartpole(
    total_updates=200,
    steps_per_update=4096,
    gamma=0.99,
    lam=0.95,
    max_kl=1e-2,
    cg_iters=10,
    damping=1e-2,
    vf_lr=1e-3,
    vf_iters=10,
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
    opt_v = optim.Adam(v.parameters(), lr=vf_lr)

    for upd in range(1, total_updates + 1):
        # collect on-policy batch and compute GAE advantages
        obs_b, act_b, logp_old_b, adv_b, ret_b = collect_trajectories(
            env, pi, v, steps_per_update, device=device, gamma=gamma, lam=lam
        )

        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_b, dtype=torch.int64, device=device)
        logp_old_t = torch.tensor(logp_old_b, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_b, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_b, dtype=torch.float32, device=device)

        # ---- TRPO policy step ----
        with torch.no_grad():
            old_dist = pi.dist(obs_t)

        dist = pi.dist(obs_t)
        logp = dist.log_prob(act_t)
        ratio = torch.exp(logp - logp_old_t)
        surrogate = (ratio * adv_t).mean()

        # gradient of surrogate
        grads = torch.autograd.grad(surrogate, pi.parameters(), create_graph=True)
        g = flat_grad(grads).detach()

        # KL( old || new ) and its Hessian-vector product (Fisher-vector product)
        def kl_fn():
            new_dist = pi.dist(obs_t)
            kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            return kl

        kl = kl_fn()
        kl_grads = torch.autograd.grad(kl, pi.parameters(), create_graph=True)
        flat_kl_grads = flat_grad(kl_grads)

        def Fvp(vv):
            # Hessian-vector product of KL
            kl_v = (flat_kl_grads * vv).sum()
            hvp = torch.autograd.grad(kl_v, pi.parameters(), retain_graph=True)
            hvp_flat = flat_grad(hvp).detach()
            return hvp_flat + damping * vv

        step_dir = conjugate_gradient(Fvp, g, iters=cg_iters)
        shs = 0.5 * (step_dir * Fvp(step_dir)).sum()
        if shs.item() <= 0:
            # fallback: if numerical issue, skip policy update
            step_size = 0.0
            full_step = torch.zeros_like(step_dir)
        else:
            step_size = torch.sqrt(max_kl / (shs + 1e-8))
            full_step = step_size * step_dir

        old_params = flat_params(pi).clone()

        # line search to ensure KL constraint + improve surrogate
        def loss_and_kl():
            new_dist = pi.dist(obs_t)
            logp_new = new_dist.log_prob(act_t)
            ratio_new = torch.exp(logp_new - logp_old_t)
            surr = (ratio_new * adv_t).mean()
            kl_val = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            return surr, kl_val

        expected_improve = (g * full_step).sum().item()
        success = False
        for backtrack in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            new_params = old_params + backtrack * full_step
            set_flat_params(pi, new_params)
            surr_new, kl_new = loss_and_kl()
            improve = (surr_new - surrogate).item()
            if (kl_new.item() <= max_kl) and (improve >= 0.1 * backtrack * expected_improve):
                success = True
                break

        if not success:
            set_flat_params(pi, old_params)  # revert if no acceptable step

        # ---- value function fit ----
        for _ in range(vf_iters):
            v_pred = v(obs_t)
            loss_v = ((v_pred - ret_t) ** 2).mean()
            opt_v.zero_grad()
            loss_v.backward()
            opt_v.step()

        if upd % 10 == 0:
            returns = []
            for _ in range(5):
                _, _, _, rews = rollout_episode(env, pi, device=device)
                returns.append(float(rews.sum()))
            print(f"[TRPO] upd={upd:4d}  avg_return_5ep={np.mean(returns):7.1f}")

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
    pi, v = train_trpo_cartpole()
    watch_policy_human(pi)