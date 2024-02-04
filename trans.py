import os
import argparse
from collections import OrderedDict
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn, optim
from torch.distributions import Categorical

from utils import *

config = {
    'model_dir': 'models',
    'model': f'actor_only.positive_reward',
    'env_id': 'PongDeterministic-v0',
    'game_visible': False,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'wandb': 'pong-actor-only',
    'wandb': None,
    'run_in_notebook': False,

    'n_envs': 2,  # simultaneous processing environments
    'seq_len': 1,
    'lr': 1e-4,
    'hidden_dim': 256,  # hidden size, linear units of the output layer
    'batch_size': 64,
    'c_entropy': 0.1,  # entropy coefficient
    'steps_to_test': 256 * 100,  # steps to test and save
    'n_tests': 10,  # run this number of tests
    'max_steps': 2000000,
    'target_reward': 20,
}
cfg = Config(**config)
print(cfg)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        feature = self.feature(x)
        probs = self.actor(feature)
        dist = Categorical(probs)
        return dist


class RoundBuffer:
    def __init__(self):
        self.history_state_rounds = []
        self.history_action_rounds = []
        self.history_reward_rounds = []
        self.pending_states = []
        self.pending_actions = []
        self.index = 0

    def append(self, state, action, reward):
        self.pending_states.append(state)
        self.pending_actions.append(action)
        if reward != 0:
            if reward > 0:
                self.history_state_rounds.append(self.pending_states)
                self.history_action_rounds.append(self.pending_actions)
                self.history_reward_rounds.append([reward] * len(self.pending_states))
            self.clear_pending()

    def clear_pending(self):
        self.pending_states = []
        self.pending_actions = []

    def get_last_n_pending_states(self, n):
        assert n >= 0
        begin = max(0, len(self.pending_states) - n)
        return self.pending_states[begin:]

    def has_data(self, seq_len):
        return len(self.history_state_rounds) > 0

    def _pop_head(self):
        self.history_state_rounds = self.history_state_rounds[1:]
        self.history_action_rounds = self.history_action_rounds[1:]
        self.history_reward_rounds = self.history_reward_rounds[1:]

    def roll_data(self, seq_len):
        assert self.has_data(seq_len)
        assert self.index < len(self.history_state_rounds[0])

        end = self.index + 1
        begin = max(0, end - seq_len)

        seq_state = self.history_state_rounds[0][begin:end]
        seq_action = self.history_action_rounds[0][begin:end]
        seq_reward = self.history_reward_rounds[0][begin:end]
        assert len(seq_state) > 0

        self.index += 1
        if self.index >= len(self.history_state_rounds[0]):
            self._pop_head()
            self.index = 0

        return seq_state, seq_action, seq_reward

class BatchRoundBuffer:
    def __init__(self, n, seq_len):
        assert n > 0
        assert seq_len > 0
        self.n = n
        self.seq_len = seq_len
        # do not use [RoundBuffer()] * n which will create n same round_buffer
        self.round_buffers = [RoundBuffer() for _ in range(n)]

    def add(self, state, action, reward, done):
        for i in range(self.n):
            self.round_buffers[i].append(state[i], action[i], reward[i])
            if done[i]:
                self.round_buffers[i].clear_pending()

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
        return sum([rb.has_data(self.seq_len) for rb in self.round_buffers]) == len(self.round_buffers)

    def next_batch_seq(self):
        states = []
        actions = []
        labels = []
        for i in range(self.n):
            state, action, reward = self.round_buffers[i].roll_data(self.seq_len)
            states.append(state)
            actions.append(action)
            labels.append(reward[-1])
        masks = self._pad_seq(states)
        self._pad_seq(actions)
        return np.array(states), np.array(actions), np.array(masks), np.array(labels)


def optimise(model, optimizer, states, actions, masks, labels):
    params_feature = model.get_parameter('feature.linear.weight')

    dist = model(states)
    actions = actions.reshape(1, len(actions))
    log_probs = dist.log_prob(actions)
    log_probs = log_probs.reshape(-1, 1)
    actor_loss = -(log_probs * labels.unsqueeze(-1)).mean()
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

def train_():
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
    best_reward = None

    brb = BatchRoundBuffer(cfg.n_envs, cfg.seq_len)
    state = envs.reset()
    state = grey_crop_resize_batch(state)

    while not early_stop and steps < cfg.max_steps:
        seq_state, seq_mask = brb.peek_state_seq(state)
        seq_state = torch.FloatTensor(seq_state).to(cfg.device)
        dist = model(seq_state)
        action = dist.sample().to(cfg.device)
        next_state, reward, done, _ = envs.step(action.cpu().numpy())
        next_state = grey_crop_resize_batch(next_state)  # simplify perceptions (grayscale-> crop-> resize) to train CNN

        brb.add(state, action.cpu().numpy(), reward, done)
        state = next_state
        steps += 1

        total_reward_1_env += reward[0]
        steps_1_env += 1
        if done[0]:
            total_runs_1_env += 1
            print(f'Run {total_runs_1_env}, steps {steps_1_env}, Reward {total_reward_1_env}')
            writer.add_scalar('Reward/train_reward_1_env', total_reward_1_env, steps)
            total_reward_1_env = 0
            steps_1_env = 0

        if brb.has_full_batch():
            states, actions, masks, labels = brb.next_batch_seq()
            result = optimise(model, optimizer,
                     torch.FloatTensor(states).to(cfg.device),
                     torch.from_numpy(actions).to(cfg.device),
                     torch.from_numpy(masks).to(cfg.device),
                     torch.from_numpy(labels).to(cfg.device))

            writer.add_scalar('Loss/Total Loss', result.loss.item(), steps)
            writer.add_scalar('Loss/Actor Loss', result.actor_loss.item(), steps)
            writer.add_scalar('Loss/Entropy Loss', result.entropy_loss.item(), steps)

        if steps % cfg.steps_to_test == 0:
            model.eval()
            test_reward = np.mean([test_env(env_test, model) for _ in range(cfg.n_tests)])  # do N_TESTS tests and takes the mean reward
            model.train()
            print('Step: %d -> Reward: %s' % (steps, test_reward))
            writer.add_scalar('Reward/test_reward', test_reward, steps)

            if best_reward is None or best_reward < test_reward:  # save a checkpoint every time it achieves a better reward
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                    name = "%s_%s_%+.3f_%d.pth" % (cfg.model, cfg.env_id, test_reward, steps)
                    fname = os.path.join(cfg.model_dir, name)
                    states = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(states, fname)
                best_reward = test_reward

            if test_reward > cfg.target_reward:
                early_stop = True


def test_env(env, model):
    state, _ = env.reset()
    state = grey_crop_resize(state)

    done = False
    total_reward = 0
    brb = BatchRoundBuffer(1, cfg.seq_len)
    while not done:
        seq_state, seq_mask = brb.peek_state_seq(state[None, :])
        seq_state = torch.FloatTensor(seq_state).to(cfg.device)
        dist = model(seq_state)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _, _ = env.step(action)
        next_state = grey_crop_resize(next_state)

        brb.add(state[None, :], np.array([action]), np.array([reward]), np.array([done]))
        state = next_state
        total_reward += reward
    return total_reward


def train(load_from=None):
    if cfg.wandb is not None:
        import wandb
        wandb_login()
        wandb.init(project=cfg.wandb, sync_tensorboard=True, config=cfg)
    train_()
    if cfg.wandb is not None:
        import wandb
        wandb.finish()

def eval(load_from=None):
    pass


if __name__ == "__main__":
    # python ppo_pong.py --eval --model models/ppo_pong.diff.5200.pth
    ap = argparse.ArgumentParser(description='Process args.')
    ap.add_argument('--eval', action='store_true', help='evaluate')
    ap.add_argument('--model', type=str, default=None, help='model to load')
    args = ap.parse_args(args=[]) if cfg.run_in_notebook else ap.parse_args()

    os.makedirs(cfg.model_dir, mode=0o644, exist_ok=True)
    if not args.eval:
        train(args.model)
    else:
        eval(args.model)
