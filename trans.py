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
    'model': 'actor_only.norm_reward',
    'env_id': 'PongDeterministic-v0',
    'game_visible': False,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'wandb': 'actor_only.norm_reward',
    'wandb': None,
    'run_in_notebook': False,

    'n_envs': 8,  # simultaneous processing environments
    'seq_len': 1,
    'lr': 1e-4,
    'hidden_dim': 256,  # hidden size, linear units of the output layer
    'batch_size': 256,
    'c_entropy': 0.0,  # entropy coefficient
    'steps_to_test': 256 * 200,  # steps to test and save
    'n_tests': 10,  # run this number of tests
    'max_steps': 2000000,
    'target_reward': 20,
}
cfg = Config(**config)
writer: MySummaryWriter = None


class SeqModel(nn.Module):
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
        self.actor = nn.Sequential(  # The “Actor” updates the policy distribution in the direction suggested by the Critic (such as with policy gradients)
            # nn.Linear(in_features=cfg.hidden_dim, out_features=cfg.hidden_dim),
            # nn.ReLU(),
            nn.Linear(in_features=cfg.hidden_dim, out_features=num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        x = x.squeeze(1)
        feature = self.feature(x)
        probs = self.actor(feature)
        dist = Categorical(probs)
        return dist


class RoundBuffer:
    def __init__(self):
        self.history_states_roundly = []
        self.history_actions_roundly = []
        self.history_rewards_roundly = []
        self.pending_states = []
        self.pending_actions = []
        self.pending_rewards = []
        self.index = 0

    def append(self, state, action, reward, done):
        self.pending_states.append(state)
        self.pending_actions.append(action)
        # deal with reward
        if reward != 0:
            l = len(self.pending_states) - len(self.pending_rewards)
            assert l > 0
            self.pending_rewards.extend([reward] * l)
        if reward != 0:
            indices = list(range(len(self.pending_states)))
            random.shuffle(indices)
            self.history_states_roundly.append([self.pending_states[i] for i in indices])
            self.history_actions_roundly.append([self.pending_actions[i] for i in indices])
            self.history_rewards_roundly.append([self.pending_rewards[i] for i in indices])
            self.clear_pending()
        if done:
            self.clear_pending()

    def clear_pending(self):
        self.pending_states = []
        self.pending_actions = []
        self.pending_rewards = []

    def get_last_n_pending_states(self, n):
        assert n >= 0
        begin = max(0, len(self.pending_states) - n)
        return self.pending_states[begin:]

    def has_data(self, count):
        assert count > 0
        return sum([len(states) for states in self.history_states_roundly]) - self.index >= count

    def _pop_head(self):
        self.history_states_roundly = self.history_states_roundly[1:]
        self.history_actions_roundly = self.history_actions_roundly[1:]
        self.history_rewards_roundly = self.history_rewards_roundly[1:]

    def roll_data(self, seq_len):
        assert self.has_data(1)
        assert self.index < len(self.history_states_roundly[0])

        end = self.index + 1
        begin = max(0, end - seq_len)

        seq_state = self.history_states_roundly[0][begin:end]
        seq_action = self.history_actions_roundly[0][begin:end]
        seq_reward = self.history_rewards_roundly[0][begin:end]
        assert len(seq_state) > 0

        self.index += 1
        if self.index >= len(self.history_states_roundly[0]):
            self._pop_head()
            self.index = 0

        return seq_state, seq_action, seq_reward

class BatchRoundBuffer:
    def __init__(self, n, batch_size, seq_len):
        assert n > 0
        assert seq_len > 0
        assert batch_size > 0
        assert batch_size % n == 0
        self.n = n
        self.batch_size = batch_size
        self.seq_len = seq_len
        # do not use [RoundBuffer()] * n which will create n same round_buffer
        self.round_buffers = [RoundBuffer() for _ in range(n)]

    def add(self, state, action, reward, done):
        for i in range(self.n):
            self.round_buffers[i].append(state[i], action[i], reward[i], done[i])

    def peek_state_seq(self, state):
        states = []
        for i in range(len(state)):
            st = self.round_buffers[i].get_last_n_pending_states(self.seq_len - 1)
            st = st + [state[i]]
            states.append(st)
        masks = self._pad_seq(states)
        return np.array(states), np.array(masks)

    def _pad_seq(self, datas):
        masks = []
        for i in range(len(datas)):
            mask_len = self.seq_len - len(datas[i])
            datas[i] += [np.zeros_like(datas[i][0]) for _ in range(mask_len)]
            mask = [False] * (self.seq_len - mask_len) + [True] * mask_len
            masks.append(mask)
        return masks

    def has_full_batch(self):
        m = self.batch_size // self.n
        return sum([rb.has_data(m) for rb in self.round_buffers]) == len(self.round_buffers)

    def next_batch_seq(self):
        m = self.batch_size // self.n
        states = []
        actions = []
        rewards = []
        for _ in range(m):
            for i in range(self.n):
                state, action, reward = self.round_buffers[i].roll_data(self.seq_len)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
        masks = self._pad_seq(states)
        self._pad_seq(actions)
        rewards = np.array(rewards)
        # rewards = self.normalize_rewards(rewards)
        return np.array(states), np.array(actions), rewards, np.array(masks)

    def normalize_rewards(self, rewards):
        return (rewards - rewards.mean()) / (rewards.std() + 1e-9)

def optimise(model, optimizer, states, actions, rewards, masks):
    params_feature = model.get_parameter('feature.linear.weight')

    dist = model(states)
    actions = actions.reshape(1, len(actions))
    log_probs = dist.log_prob(actions)
    log_probs = log_probs.reshape(-1, 1)
    actor_loss = -(log_probs * rewards.unsqueeze(-1)).mean()
    entropy_loss = -dist.entropy().mean()
    loss = actor_loss + cfg.c_entropy * entropy_loss

    if not writer.check_steps():
        optimizer.zero_grad()  # in PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        loss.backward()  # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
        optimizer.step()  # performs the parameters update based on the current gradient and the update rule
    else:
        optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        grad_actor = params_feature.grad.mean(), params_feature.grad.std()

        optimizer.zero_grad()
        (cfg.c_entropy * entropy_loss).backward(retain_graph=True)
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
    writer = MySummaryWriter(0, cfg.steps_to_test, comment=f'.{cfg.model}.{cfg.env_id}')

    envs = [lambda: gym.make(cfg.env_id, render_mode='human' if cfg.game_visible else 'rgb_array')] * cfg.n_envs  # Prepare N actors in N environments
    envs = SubprocVecEnv(envs)  # Vectorized Environments are a method for stacking multiple independent environments into a single environment. Instead of the training an RL agent on 1 environment per step, it allows us to train it on n environments per step. Because of this, actions passed to the environment are now a vector (of dimension n). It is the same for observations, rewards and end of episode signals (dones). In the case of non-array observation spaces such as Dict or Tuple, where different sub-spaces may have different shapes, the sub-observations are vectors (of dimension n).
    num_outputs = envs.action_space.n

    env_test = gym.make(cfg.env_id, render_mode='rgb_array')

    model = SeqModel(cfg, num_outputs).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)  # implements Adam algorithm

    early_stop = False
    total_reward_1_env = 0
    total_runs_1_env = 0
    steps_1_env = 0
    steps = 0
    best_reward = -np.Inf

    if load_from is not None:
        checkpoint = torch.load(load_from, map_location=cfg.device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = checkpoint['global_steps']
        best_reward = checkpoint['best_reward']
        print(f'Model loaded, starting from steps {steps}')
        print('Previous best reward: %.3f' % best_reward)

    print(cfg)
    print(model)
    print(optimizer)

    brb = BatchRoundBuffer(cfg.n_envs, cfg.batch_size, cfg.seq_len)
    state = envs.reset()
    state = grey_crop_resize_batch(state)

    while not early_stop and steps < cfg.max_steps:
        seq_state, seq_mask = brb.peek_state_seq(state)
        seq_state = torch.FloatTensor(seq_state).to(cfg.device)
        dist = model(seq_state)
        action = dist.sample().to(cfg.device)
        # action = torch.argmax(dist.probs, dim=1).to(cfg.device)
        next_state, reward, done, _ = envs.step(action.cpu().numpy())
        next_state = grey_crop_resize_batch(next_state)  # simplify perceptions (grayscale-> crop-> resize) to train CNN

        brb.add(state, action.cpu().numpy(), reward, done)
        state = next_state
        steps += 1
        writer.update_global_step(steps)

        total_reward_1_env += reward[0]
        steps_1_env += 1
        if done[0]:
            total_runs_1_env += 1
            print(f'Run {total_runs_1_env}, steps {steps_1_env}, Reward {total_reward_1_env}')
            writer.add_scalar('Reward/train_reward_1_env', total_reward_1_env, steps)
            total_reward_1_env = 0
            steps_1_env = 0

        if brb.has_full_batch():
            states, actions, rewards, masks = brb.next_batch_seq()
            result = optimise(model, optimizer,
                     torch.FloatTensor(states).to(cfg.device),
                     torch.from_numpy(actions).to(cfg.device),
                     torch.from_numpy(rewards).to(cfg.device),
                     torch.from_numpy(masks).to(cfg.device))

            writer.add_scalar('Loss/Total Loss', result.loss.item(), steps)
            writer.add_scalar('Loss/Actor Loss', result.actor_loss.item(), steps)
            writer.add_scalar('Loss/Entropy Loss', result.entropy_loss.item(), steps)

        if steps % cfg.steps_to_test == 0:
            model.eval()
            test_reward = np.mean([test_env(env_test, model) for _ in range(cfg.n_tests)])  # do N_TESTS tests and takes the mean reward
            model.train()
            print('Step: %d -> Reward: %s' % (steps, test_reward))
            writer.add_scalar('Reward/test_reward', test_reward, steps)

            if best_reward < test_reward:  # save a checkpoint every time it achieves a better reward
                print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                best_reward = test_reward
                name = "%s_%s_%+.3f_%d.pth" % (cfg.model, cfg.env_id, test_reward, steps)
                fname = os.path.join(cfg.model_dir, name)
                states = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_reward': best_reward,
                    'global_steps': steps,
                }
                torch.save(states, fname)

            if test_reward > cfg.target_reward:
                early_stop = True


def test_env(env, model):
    state, _ = env.reset()
    state = grey_crop_resize(state)

    done = False
    total_reward = 0
    brb = BatchRoundBuffer(1, 1, cfg.seq_len)
    while not done:
        seq_state, seq_mask = brb.peek_state_seq(state[None, :])
        seq_state = torch.FloatTensor(seq_state).to(cfg.device)
        dist = model(seq_state)
        action = dist.sample().cpu().numpy()[0]
        # action = torch.argmax(dist.probs).to(cfg.device)
        next_state, reward, done, _, _ = env.step(action)
        next_state = grey_crop_resize(next_state)

        brb.add(state[None, :], np.array([action]), np.array([reward]), np.array([done]))
        state = next_state
        total_reward += reward
    return total_reward


def train(load_from=None):
    wandb_init(cfg.wandb, cfg)
    train_(load_from)
    wandb_finish(cfg.wandb)

def eval(load_from=None):
    pass


if __name__ == "__main__":
    # python ppo_pong.py --eval --model models/ppo_pong.diff.5200.pth
    ap = argparse.ArgumentParser(description='Process args.')
    ap.add_argument('--eval', action='store_true', help='evaluate')
    ap.add_argument('--model', type=str, default=None, help='model to load')
    args = ap.parse_args(args=[]) if cfg.run_in_notebook else ap.parse_args()

    os.makedirs(cfg.model_dir, mode=0o755, exist_ok=True)
    if not args.eval:
        train(args.model)
    else:
        eval(args.model)
