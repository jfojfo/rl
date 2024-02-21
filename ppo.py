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
    'model_net': 'cnn3d',  # mlp, cnn, cnn3d
    'model': 'pg.episode.ppo.cnn3d',
    'model_dir': 'models',
    'env_id': 'PongDeterministic-v0',
    'game_visible': False,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'wandb': 'pg.episode.ppo',
    # 'run_in_notebook': True,
    'wandb': None,
    'run_in_notebook': False,

    'n_envs': 8,  # simultaneous processing environments
    'lr': 1e-4,
    'hidden_dim': 200,  # hidden size, linear units of the output layer
    'c_entropy': 0.0,  # entropy coefficient
    'gamma': 0.99,
    'surr_clip': 0.2,

    'optimise_times': 10,  # optimise times per epoch
    'max_batch_size': 3000,  # mlp 10w, cnn 3w, cnn3d 3k
    'chunk_percent': 1/64,  # split into chunks, do optimise per chunk
    'seq_len': 11,
    'epoch_episodes': 10,
    'epoch_save': 20,
    'max_epoch': 1000000,
    'target_reward': 20,
    'diff_state': False,
    'shuffle': True,
}
cfg = Config(**config)
writer: MySummaryWriter = None


def get_model():
    if cfg.model_net == 'mlp':
        return MLPModel
    elif cfg.model_net == 'cnn':
        return CNNModel
    elif cfg.model_net == 'cnn3d':
        return CNN3dModel
    elif cfg.model_net == 'cnn3d2d':
        return CNN3d2dModel
    else:
        return MLPModel


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
        self.critic = nn.Sequential(
            nn.Linear(in_features=cfg.hidden_dim, out_features=1),
        )

    def forward(self, x: torch.Tensor):
        feature = self.feature(x)
        logits = self.actor(feature)
        dist = Categorical(logits=logits)
        value = self.critic(feature)
        return dist, value


class CNNModel(nn.Module):
    def __init__(self, cfg: Config, num_outputs) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4, padding=2)),
            ('act1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('act2', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=32 * 10 * 10, out_features=cfg.hidden_dim)),
        ]))
        self.actor = nn.Sequential(
            # nn.Linear(in_features=cfg.hidden_dim, out_features=cfg.hidden_dim),
            # nn.ReLU(),
            nn.Linear(in_features=cfg.hidden_dim, out_features=num_outputs),
        )
        self.critic = nn.Sequential(
            nn.Linear(in_features=cfg.hidden_dim, out_features=1),
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        feature = self.feature(x)
        logits = self.actor(feature)
        dist = Categorical(logits=logits)
        value = self.critic(feature)
        return dist, value


class CNN3dModel(nn.Module):
    def __init__(self, cfg: Config, num_outputs) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 8, 8), stride=(2, 4, 4), padding=(0, 2, 2))),
            ('act1', nn.ReLU()),
            ('conv2', nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 4, 4), stride=2, padding=(0, 1, 1))),
            ('act2', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=32 * 2 * 10 * 10, out_features=cfg.hidden_dim)),
        ]))
        self.actor = nn.Sequential(
            nn.Linear(in_features=cfg.hidden_dim, out_features=num_outputs),
        )
        self.critic = nn.Sequential(
            nn.Linear(in_features=cfg.hidden_dim, out_features=1),
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 4, 1, 2, 3)
        feature = self.feature(x)
        logits = self.actor(feature)
        dist = Categorical(logits=logits)
        value = self.critic(feature)
        return dist, value


class CNN3d2dModel(nn.Module):
    def __init__(self, cfg: Config, num_outputs) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature3d = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 8, 8), stride=(2, 4, 4), padding=(0, 2, 2))),
            ('act1', nn.ReLU()),
            ('conv2', nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 4, 4), stride=2, padding=(0, 1, 1))),
            ('act2', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=32 * 2 * 10 * 10, out_features=cfg.hidden_dim)),
        ]))
        self.feature2d = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4, padding=2)),
            ('act1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('act2', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=32 * 10 * 10, out_features=cfg.hidden_dim)),
        ]))
        self.actor = nn.Sequential(
            # nn.Linear(in_features=cfg.hidden_dim, out_features=cfg.hidden_dim),
            # nn.ReLU(),
            nn.Linear(in_features=cfg.hidden_dim * 2, out_features=num_outputs),
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 4, 1, 2, 3)
        feature3d = self.feature3d(x)
        feature2d = self.feature2d(x[:, :, -1, ...])
        feature = torch.cat([feature3d, feature2d], dim=-1)
        logits = self.actor(feature)
        dist = Categorical(logits=logits)
        return dist


class EpisodeCollector:
    def __init__(self):
        self.episodes = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.extras = []

    def add(self, state, action, reward, done, *extra_pt_tensors):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(bool(done))  # prevent torch.as_tensor throwing warning for np.bool_ array
        self.extras.append(extra_pt_tensors)
        if done:
            assert len(self.actions) > 0
            self.episodes.append([self.states, self.actions, self.rewards, self.dones, *zip(*self.extras)])
            self.clear_pending()

    def roll(self):
        if len(self.episodes) > 0:
            return self.episodes.pop(0)
        return None

    def peek(self, seq_len):
        assert seq_len > 0
        start = max(0, len(self.actions) - seq_len)
        return self.states[start:], self.actions[start:], self.rewards[start:], self.dones[start:], *zip(*self.extras[start:])

    def clear_pending(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
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

    def peek_batch(self, seq_len, state):
        new_state = []
        for i in range(self.n):
            seq_state, *_ = self.episode_collectors[i].peek(seq_len)
            d = seq_len - len(seq_state)
            if d > 0:
                seq_state = [state[i] - state[i]] * d + seq_state
            assert len(seq_state) == seq_len
            seq_state.append(state[i])
            new_state.append(seq_state)
        return np.array(new_state)


class BaseGenerator:
    # (s1, s2, ...), (a1, a2, ...), (r1, r2, ...), ...
    def __init__(self, batch_size, data_fn, random=False, *data):
        self.batch_size = batch_size
        self.data_fn = data_fn
        self.random = random
        self.data = list(data)

    def __len__(self):
        return len(self.data[0]) if len(self.data) > 0 else 0

    def shuffle(self):
        indices = list(range(len(self)))
        random.shuffle(indices)
        for i, d in enumerate(self.data):
            self.data[i] = [d[i] for i in indices]

    def next_batch(self):
        if self.random:
            self.shuffle()
        for i in range(0, len(self), self.batch_size):
            yield self.data_fn(*[d[i:i + self.batch_size] for d in self.data])


class EpDataGenerator(BaseGenerator):
    def __init__(self, episodes, batch_size, data_fn, random=False):
        states, actions, rewards, dones = [], [], [], []
        extra = []
        round_count = 0
        for episode in episodes:
            ep_states, ep_actions, ep_rewards, ep_dones, *ep_extra = episode
            t = np.array(ep_rewards)
            round_count += np.sum(t == 1) + np.sum(t == -1)
            states.extend(ep_states)
            actions.extend(ep_actions)
            rewards.extend(ep_rewards)
            dones.extend(ep_dones)
            if len(extra) == 0:
                extra = [[]] * len(ep_extra)
            extra = [extra[i] + list(e) for i, e in enumerate(ep_extra)]
        self.round_count = round_count
        data = [states, actions, rewards, dones, *extra]
        super().__init__(batch_size, data_fn, random, *data)


class StateSeqEpDataGenerator(EpDataGenerator):
    def __init__(self, episodes, batch_size, seq_len, data_fn, random=False):
        super().__init__(episodes, batch_size, data_fn, random)
        self.seq_len = seq_len
        new_states = []
        states = self.data[0]
        dones = self.data[3]
        last_done_index = -1
        for i, done in enumerate(dones):
            end = i + 1
            j = end - seq_len
            start = max(0, j)
            seq_state = states[start:end]
            zero_state = seq_state[-1] - seq_state[-1]
            # mask zeroes
            if last_done_index >= start:
                count = last_done_index - start + 1
                seq_state[:count] = [zero_state] * count
            if j < 0:
                seq_state = [zero_state] * (-j) + seq_state
            # update after mask to prevent mask all
            if done:
                last_done_index = i
            new_states.append(seq_state)
            assert len(seq_state) == seq_len
        self.data[0] = new_states


def data_fn(states, actions, rewards, *extra):
    states = torch.as_tensor(np.array(states), dtype=torch.float32).to(cfg.device)  # requires_grad=False
    actions = torch.as_tensor(actions).to(cfg.device)  # requires_grad=False
    rewards = torch.as_tensor(rewards).to(cfg.device)  # requires_grad=False
    extra = [torch.as_tensor(e).to(cfg.device) for e in extra]
    return states, actions, rewards, *extra


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


def normalize(data):
    return (data - data.mean()) / (data.std() + 1e-8)


def loss_pg(model, data_loader, states, actions, rewards):
    dist, values = model(states)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    values = values.squeeze(1)

    advantages = rewards - values.detach()
    # do not normalize if random shuffled
    # advantages = normalize_rewards(advantages)
    # cum_values = values * 0.99 + rewards * 0.01

    critic_loss = 0.5 * (values - rewards).pow(2).sum() / len(data_loader)
    actor_loss = -(log_probs * advantages).sum() / data_loader.round_count
    entropy_loss = -entropy.sum() / len(data_loader)
    loss = actor_loss + critic_loss + entropy_loss * cfg.c_entropy
    return critic_loss, actor_loss, entropy_loss, loss


def loss_ppo(model, data_loader, states, actions, rewards, old_log_probs):
    dist, values = model(states)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    values = values.squeeze(1)

    # todo: normalize?
    advantages = rewards - values.detach()
    # advantages = normalize(advantages)
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - cfg.surr_clip, 1.0 + cfg.surr_clip) * advantages
    actor_loss = -torch.min(surr1, surr2).sum() / len(data_loader)

    critic_loss = 0.5 * (values - rewards).pow(2).sum() / len(data_loader)
    entropy_loss = -entropy.sum() / len(data_loader)
    loss = actor_loss + critic_loss + entropy_loss * cfg.c_entropy
    return critic_loss, actor_loss, entropy_loss, loss


def optimise_by_minibatch(model, optimizer, data_loader):
    first_iter = True
    acc_loss, acc_critic_loss, acc_actor_loss, acc_entropy_loss = 0, 0, 0, 0
    optimizer.zero_grad()
    for states, actions, rewards, dones, *extra in data_loader.next_batch():
        old_log_probes = extra[0]
        critic_loss, actor_loss, entropy_loss, loss = loss_ppo(model, data_loader, states, actions, rewards, old_log_probes)

        # summary in the first iteration and zero_grad after that
        if first_iter:
            params_feature = model.get_parameter('feature.linear.weight')
            writer.summary_grad(optimizer, params_feature, {'critic': critic_loss, 'actor': actor_loss, 'entropy': entropy_loss, 'total': loss})
            optimizer.zero_grad()
            first_iter = False

        loss.backward()

        acc_loss += loss
        acc_critic_loss += critic_loss
        acc_actor_loss += actor_loss
        acc_entropy_loss += entropy_loss
    optimizer.step()
    writer.summary_loss({'critic': acc_critic_loss, 'actor': acc_actor_loss, 'entropy': acc_entropy_loss, 'total': acc_loss})


# def optimise(model, optimizer, data_loader, batch_round_count):
#     actor_loss = 0
#     entropy_loss = 0
#     for states, actions, rewards, *_ in data_loader.next_batch():
#         dist = model(states)
#         log_prob = dist.log_prob(actions)
#         entropy = dist.entropy()

#         actor_loss += -(log_prob * rewards).sum()
#         entropy_loss += -entropy.sum()
#     # actor_loss /= len(data_loader)
#     # mean(sum(each round))
#     actor_loss /= batch_round_count
#     entropy_loss /= len(data_loader)

#     return optimise_(model, optimizer, actor_loss, entropy_loss)


# def optimise_(model, optimizer, *losses):
#     params_feature = model.get_parameter('feature.linear.weight')

#     actor_loss, entropy_loss = losses
#     loss = actor_loss + entropy_loss * cfg.c_entropy

#     if not writer.check_steps():
#         optimizer.zero_grad()  # in PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
#         loss.backward()  # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
#         optimizer.step()  # performs the parameters update based on the current gradient and the update rule
#     else:
#         optimizer.zero_grad()
#         actor_loss.backward(retain_graph=True)
#         grad_actor = params_feature.grad.mean(), params_feature.grad.std()

#         optimizer.zero_grad()
#         (entropy_loss * cfg.c_entropy).backward(retain_graph=True)
#         grad_entropy = params_feature.grad.mean(), params_feature.grad.std()

#         optimizer.zero_grad()
#         loss.backward()
#         grad_total = params_feature.grad.mean(), params_feature.grad.std()
#         grad_max = params_feature.grad.abs().max()

#         optimizer.step()

#         writer.add_scalar('Grad/Actor', grad_actor[0].item(), writer.global_step)
#         writer.add_scalar('Grad/Entropy', grad_entropy[0].item(), writer.global_step)
#         writer.add_scalar('Grad/Total', grad_total[0].item(), writer.global_step)
#         writer.add_scalar('Grad/Max', grad_max.item(), writer.global_step)

#     return Config(**{
#         'loss': loss,
#         'actor_loss': actor_loss,
#         'entropy_loss': entropy_loss
#     })


def diff_state(state, last_state):
    return state if not cfg.diff_state else (state - last_state)


def train_(load_from):
    global writer
    writer = MySummaryWriter(0, cfg.epoch_save // 4, comment=f'.{cfg.model}.{cfg.env_id}')
    writer.summary_script_content(__file__)

    envs = [lambda: gym.make(cfg.env_id, render_mode='human' if cfg.game_visible else 'rgb_array')] * cfg.n_envs  # Prepare N actors in N environments
    envs = SubprocVecEnv(envs)  # Vectorized Environments are a method for stacking multiple independent environments into a single environment. Instead of the training an RL agent on 1 environment per step, it allows us to train it on n environments per step. Because of this, actions passed to the environment are now a vector (of dimension n). It is the same for observations, rewards and end of episode signals (dones). In the case of non-array observation spaces such as Dict or Tuple, where different sub-spaces may have different shapes, the sub-observations are vectors (of dimension n).
    num_outputs = 2  #envs.action_space.n

    model = get_model()(cfg, num_outputs).to(cfg.device)
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
        state_ = diff_state(state, last_state)
        if cfg.model_net in ('cnn3d', 'cnn3d2d'):
            state_ = collector.peek_batch(cfg.seq_len - 1, state_)
        dist, value = model(torch.FloatTensor(state_).to(cfg.device))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.cpu().numpy()
        next_state, reward, done, _ = envs.step(action + 2)
        next_state = batch_prepro(next_state)  # simplify perceptions (grayscale-> crop-> resize) to train CNN

        collector.add(diff_state(state, last_state), action, reward, done, log_prob.detach())
        last_state = state
        state = next_state
        # reset last_state to state when done
        for i, d in enumerate(done):
            if d:
                last_state[i] = state[i]

        if collector.has_full_batch(cfg.epoch_episodes):
            epoch += 1
            writer.update_global_step(epoch)
            avg_reward = 0

            # [(states, actions, rewards, ...), ...]
            episodes = collector.roll_batch()
            total_samples = 0
            for i, episode in enumerate(episodes):
                ep_states, ep_actions, ep_rewards, ep_dones, *_ = episode
                ep_reward_sum, ep_steps = sum(ep_rewards), len(ep_rewards)
                avg_reward += ep_reward_sum
                cum_reward = ep_reward_sum if cum_reward is None else cum_reward * 0.99 + ep_reward_sum * 0.01
                print(f'Run {(epoch - 1) * cfg.epoch_episodes + i + 1}, steps {ep_steps}, reward {ep_reward_sum}, cum_reward {cum_reward:.3f}')

                ep_rewards = np.array(ep_rewards)
                ep_rewards = discount_rewards(ep_rewards, cfg.gamma)
                # ep_rewards = normalize(ep_rewards)
                ep_rewards = list(ep_rewards)
                episode[2] = ep_rewards
                total_samples += len(ep_rewards)
            avg_reward /= len(episodes)
            writer.add_scalar('Reward/epoch_reward_avg', avg_reward, epoch)

            for _ in range(cfg.optimise_times):
                # optimise every chunk_percent samples to accelerate training
                chunk_size = int(np.ceil(total_samples * cfg.chunk_percent))
                # set random True
                if cfg.model_net in ('cnn3d', 'cnn3d2d'):
                    chunk_loader = StateSeqEpDataGenerator(episodes, chunk_size, cfg.seq_len, lambda *d: d, random=cfg.shuffle)
                else:
                    chunk_loader = EpDataGenerator(episodes, chunk_size, lambda *d: d, random=cfg.shuffle)
                for chunk in chunk_loader.next_batch():
                    # minibatch don't exceeds cfg.max_batch_size
                    minibatch_loader = BaseGenerator(cfg.max_batch_size, data_fn, False, *chunk)
                    optimise_by_minibatch(model, optimizer, minibatch_loader)
            # write summary once after optimise_times to prevent duplicate
            writer.write_summary()

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

    collector = BatchEpisodeCollector(1)
    done = False
    total_reward = 0
    steps = 0
    while not done:
        state_ = diff_state(state, last_state)
        if cfg.model_net in ('cnn3d', 'cnn3d2d'):
            state_ = collector.peek_batch(cfg.seq_len - 1, [state_])
        dist, value = model(torch.FloatTensor(state_).to(cfg.device))
        # dist = model(torch.FloatTensor(diff_state(state, last_state)).unsqueeze(0).to(cfg.device))
        action = dist.sample()
        action = action.cpu().numpy()[0]
        next_state, reward, done, _, _ = env.step(action + 2)
        next_state = prepro(next_state)

        collector.add([diff_state(state, last_state)], [action], [reward], [done])
        last_state = state
        state = next_state
        total_reward += reward
        steps += 1
    collector.clear_all()
    return total_reward, steps


def train(load_from=None):
    wandb_init(cfg.wandb, cfg)
    train_(load_from)
    wandb_finish(cfg.wandb)

def eval(load_from=None):
    assert load_from is not None

    env_test = gym.make(cfg.env_id, render_mode='human')
    # num_outputs = env_test.action_space.n
    num_outputs = 2
    model = get_model()(cfg, num_outputs).to(cfg.device)
    model.eval()

    checkpoint = torch.load(load_from, map_location=cfg.device)
    model.load_state_dict(checkpoint['state_dict'])

    while True:
        reward, steps = test_env(env_test, model)
        print(f"steps {steps}, reward {reward}")


def test_state_seq_data_generator():
    episode = [
        [i+1 for i in range(10)],
        [i+1 for i in range(10)],
        [i+1 for i in range(10)],
        [False, False, True, False, False, True, False, False, False, True],
    ]
    g = StateSeqEpDataGenerator([episode] * 2, 2, 3, data_fn)
    assert g.data[0] == [[0, 0, 1], [0, 1, 2], [1, 2, 3], [0, 0, 4], [0, 4, 5], [4, 5, 6], [0, 0, 7], [0, 7, 8], [7, 8, 9], [8, 9, 10], [0, 0, 1], [0, 1, 2], [1, 2, 3], [0, 0, 4], [0, 4, 5], [4, 5, 6], [0, 0, 7], [0, 7, 8], [7, 8, 9], [8, 9, 10]]


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
