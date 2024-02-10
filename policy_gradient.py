import os
import argparse
from collections import OrderedDict
import random
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn, optim
from torch.distributions import Categorical

from utils import *

config = {
    'model_dir': 'models',
    'model': 'pg.episode.norm_discount_reward.div_round_count',
    'env_id': 'PongDeterministic-v0',
    'game_visible': False,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'wandb': 'actor_only.norm_reward',
    'wandb': None,
    'run_in_notebook': False,

    'n_envs': 8,  # simultaneous processing environments
    'lr': 1e-4,
    'hidden_dim': 200,  # hidden size, linear units of the output layer
    'c_entropy': 0.0,  # entropy coefficient
    'gamma': 0.99,

    'max_batch_size': 100000,  # mlp 10w, cnn 3w
    'epoch_episodes': 10,
    'epoch_save': 500,
    'max_epoch': 1000000,
    'target_reward': 20,
}
cfg = Config(**config)
writer: MySummaryWriter = None


class MLPModel(nn.Module):
    def __init__(self, cfg: Config, num_outputs) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=80 * 80, out_features=cfg.hidden_dim)),
        ]))
        self.actor = nn.Sequential(
            nn.Linear(in_features=cfg.hidden_dim, out_features=num_outputs),
        )

    def forward(self, x: torch.Tensor):
        feature = self.feature(x)
        logits = self.actor(feature)
        dist = Categorical(logits=logits)
        return dist


class CNNModel(nn.Module):
    def __init__(self, cfg: Config, num_outputs) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4)),
            ('act1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)),
            ('act2', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=32 * 9 * 9, out_features=cfg.hidden_dim)),
        ]))
        self.actor = nn.Sequential(
            # nn.Linear(in_features=cfg.hidden_dim, out_features=cfg.hidden_dim),
            # nn.ReLU(),
            nn.Linear(in_features=cfg.hidden_dim, out_features=num_outputs),
        )

    def forward(self, x: torch.Tensor):
        feature = self.feature(x)
        logits = self.actor(feature)
        dist = Categorical(logits=logits)
        return dist


class EpisodeCollector:
    def __init__(self):
        self.episodes = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.extras = []

    def add(self, state, action, reward, done, *extra_pt_tensors):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.extras.append(extra_pt_tensors)
        if done:
            assert len(self.actions) > 0
            self.episodes.append([self.states, self.actions, self.rewards, *zip(*self.extras)])
            self.clear_pending()

    def roll(self):
        if len(self.episodes) > 0:
            return self.episodes.pop(0)
        return None

    def clear_pending(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.extras = []

    def has_n_episodes(self, n):
        return len(self.episodes) >= n

    def episodes_count(self):
        return len(self.episodes)

    def clear_episodes(self):
        self.episodes = []


class BatchEpisodeCollector:
    def __init__(self, n):
        assert n > 0
        self.n = n
        # do not use [EpisodeCollector()] * n which will create n same objects
        self.episode_collectors = [EpisodeCollector() for _ in range(n)]

    def add(self, state, action, reward, done, *extra_pt_tensor):
        for i in range(self.n):
            extra = [pt[i] for pt in extra_pt_tensor]
            self.episode_collectors[i].add(state[i], action[i], reward[i], done[i], *extra)

    def has_full_batch(self, episodes_count):
        assert episodes_count > 0
        return sum([ec.episodes_count() for ec in self.episode_collectors]) >= episodes_count

    def roll_batch(self):
        results = []
        for i in range(self.n):
            # ep_extras: [(pt1,pt2,pt3...), (tt1,tt2,tt3,...), ...]
            while self.episode_collectors[i].episodes_count() > 0:
                results.append(self.episode_collectors[i].roll())
        return results

    def clear_all(self):
        for i in range(self.n):
            self.episode_collectors[i].clear_pending()
            self.episode_collectors[i].clear_episodes()


class DataGenerator:
    def __init__(self, batch_size, data_fn, *data):
        self.batch_size = batch_size
        self.data_fn = data_fn
        self.data = data

    def __len__(self):
        return len(self.data[0]) if len(self.data) > 0 else 0

    def next_batch(self):
        for i in range(0, len(self), self.batch_size):
            yield self.data_fn(*[d[i:i + self.batch_size] for d in self.data])


def data_fn(states, actions, rewards):
    states = torch.as_tensor(np.array(states), dtype=torch.float32).to(cfg.device)  # requires_grad=False
    actions = torch.as_tensor(actions).to(cfg.device)  # requires_grad=False
    rewards = torch.as_tensor(rewards).to(cfg.device)  # requires_grad=False
    return states, actions, rewards


def discount_rewards(rewards, gamma=1.0):
    # discounted_rewards = [0] * len(rewards)
    discounted_rewards = np.zeros_like(rewards)
    cum_reward = 0
    for i in reversed(range(len(rewards))):
        if rewards[i] != 0:
            cum_reward = 0  # reset the sum, since this was a game boundary (pong specific!)
        cum_reward = rewards[i] + gamma * cum_reward
        discounted_rewards[i] = cum_reward
    return discounted_rewards


def normalize_rewards(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


# episodes: [(states, actions, rewards, log_probs, entropies), ...]
def optimise(model, optimizer, data_loader, batch_round_count):
    params_feature = model.get_parameter('feature.linear.weight')

    actor_loss = 0
    entropy_loss = 0
    for states, actions, rewards in data_loader.next_batch():
        dist = model(states)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        actor_loss += -(log_prob * rewards).sum()
        entropy_loss += -entropy.sum()
    # actor_loss /= len(data_loader)
    # mean(sum(each round))
    actor_loss /= batch_round_count
    entropy_loss /= len(data_loader)
    loss = actor_loss + entropy_loss * cfg.c_entropy

    if not writer.check_steps():
        optimizer.zero_grad()  # in PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        loss.backward()  # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
        optimizer.step()  # performs the parameters update based on the current gradient and the update rule
    else:
        optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        grad_actor = params_feature.grad.mean(), params_feature.grad.std()

        optimizer.zero_grad()
        (entropy_loss * cfg.c_entropy).backward(retain_graph=True)
        grad_entropy = params_feature.grad.mean(), params_feature.grad.std()

        optimizer.zero_grad()
        loss.backward()
        grad_total = params_feature.grad.mean(), params_feature.grad.std()
        grad_max = params_feature.grad.abs().max()

        optimizer.step()

        writer.add_scalar('Grad/Actor', grad_actor[0].item(), writer.global_step)
        writer.add_scalar('Grad/Entropy', grad_entropy[0].item(), writer.global_step)
        writer.add_scalar('Grad/Total', grad_total[0].item(), writer.global_step)
        writer.add_scalar('Grad/Max', grad_max.item(), writer.global_step)

    return Config(**{
        'loss': loss,
        'actor_loss': actor_loss,
        'entropy_loss': entropy_loss
    })


def train_(load_from):
    global writer
    writer = MySummaryWriter(0, 50, comment=f'.{cfg.model}.{cfg.env_id}')

    envs = [lambda: gym.make(cfg.env_id, render_mode='human' if cfg.game_visible else 'rgb_array')] * cfg.n_envs  # Prepare N actors in N environments
    envs = SubprocVecEnv(envs)  # Vectorized Environments are a method for stacking multiple independent environments into a single environment. Instead of the training an RL agent on 1 environment per step, it allows us to train it on n environments per step. Because of this, actions passed to the environment are now a vector (of dimension n). It is the same for observations, rewards and end of episode signals (dones). In the case of non-array observation spaces such as Dict or Tuple, where different sub-spaces may have different shapes, the sub-observations are vectors (of dimension n).
    num_outputs = 2  #envs.action_space.n

    model = MLPModel(cfg, num_outputs).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)  # implements Adam algorithm

    early_stop = False
    epoch = 0
    cum_reward = None

    if load_from is not None:
        checkpoint = torch.load(load_from, map_location=cfg.device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f'Model loaded, starting from epoch {epoch}')

    print(cfg)
    print(model)
    print(optimizer)

    collector = BatchEpisodeCollector(cfg.n_envs)
    state = envs.reset()
    state = batch_prepro(state)
    last_state = state.copy()

    while not early_stop and epoch < cfg.max_epoch:
        dist = model(torch.FloatTensor(state - last_state).to(cfg.device))
        action = dist.sample()
        action = action.cpu().numpy()
        next_state, reward, done, _ = envs.step(action + 2)
        next_state = batch_prepro(next_state)  # simplify perceptions (grayscale-> crop-> resize) to train CNN

        collector.add(state - last_state, action, reward, done)
        last_state = state
        state = next_state

        if collector.has_full_batch(cfg.epoch_episodes):
            epoch += 1
            writer.update_global_step(epoch)
            avg_reward = 0

            # [(states, actions, rewards, ...), ...]
            episodes = collector.roll_batch()
            states, actions, rewards = [], [], []
            batch_round_count = 0
            for i, episode in enumerate(episodes):
                ep_states, ep_actions, ep_rewards = episode
                ep_reward_sum, ep_steps = sum(ep_rewards), len(ep_rewards)
                avg_reward += ep_reward_sum
                cum_reward = ep_reward_sum if cum_reward is None else cum_reward * 0.99 + ep_reward_sum * 0.01
                print(f'Run {(epoch - 1) * cfg.epoch_episodes + i + 1}, steps {ep_steps}, reward {ep_reward_sum}, cum_reward {cum_reward:.3f}')

                ep_rewards = np.array(ep_rewards)
                ep_round_count = (ep_rewards != 0).sum()
                ep_rewards = discount_rewards(ep_rewards, cfg.gamma)
                ep_rewards = list(normalize_rewards(ep_rewards))
                states.extend(ep_states)
                actions.extend(ep_actions)
                rewards.extend(ep_rewards)
                batch_round_count += ep_round_count
            avg_reward /= len(episodes)

            data_loader = DataGenerator(cfg.max_batch_size, data_fn, states, actions, rewards)
            result = optimise(model, optimizer, data_loader, batch_round_count)

            writer.add_scalar('Loss/Total Loss', result.loss.item(), epoch)
            writer.add_scalar('Loss/Actor Loss', result.actor_loss.item(), epoch)
            writer.add_scalar('Loss/Entropy Loss', result.entropy_loss.item(), epoch)
            writer.add_scalar('Reward/epoch_reward_avg', avg_reward, epoch)

            state = envs.reset()
            state = batch_prepro(state)
            last_state = state.copy()
            collector.clear_all()

            if epoch % cfg.epoch_save == 0:
                name = "%s_%s_%+.3f_%d.pth" % (cfg.model, cfg.env_id, cum_reward, epoch)
                fname = os.path.join(cfg.model_dir, cfg.model, name)
                states = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                torch.save(states, fname)

            if avg_reward > cfg.target_reward:
                early_stop = True


def test_env(env, model):
    state, _ = env.reset()
    state = prepro(state)
    last_state = state.copy()

    done = False
    total_reward = 0
    while not done:
        dist = model(torch.FloatTensor(state - last_state).unsqueeze(0).to(cfg.device))
        action = dist.sample()
        action = action.cpu().numpy()[0]
        next_state, reward, done, _, _ = env.step(action + 2)
        next_state = prepro(next_state)
        last_state = state
        state = next_state
        total_reward += reward
    return total_reward


def train(load_from=None):
    wandb_init(cfg.wandb, cfg)
    train_(load_from)
    wandb_finish(cfg.wandb)

def eval(load_from=None):
    assert load_from is not None

    env_test = gym.make(cfg.env_id, render_mode='human')
    # num_outputs = env_test.action_space.n
    num_outputs = 2
    model = MLPModel(cfg, num_outputs).to(cfg.device)
    model.eval()

    checkpoint = torch.load(load_from, map_location=cfg.device)
    model.load_state_dict(checkpoint['state_dict'])

    while True:
        reward = test_env(env_test, model)
        print(f"Reward {reward}")


if __name__ == "__main__":
    # python ppo_pong.py --eval --model models/ppo_pong.diff.5200.pth
    ap = argparse.ArgumentParser(description='Process args.')
    ap.add_argument('--eval', action='store_true', help='evaluate')
    ap.add_argument('--model', type=str, default=None, help='model to load')
    args = ap.parse_args(args=[]) if cfg.run_in_notebook else ap.parse_args()

    os.makedirs(os.path.join(cfg.model_dir, cfg.model), mode=0o755, exist_ok=True)
    if not args.eval:
        train(args.model)
    else:
        eval(args.model)
