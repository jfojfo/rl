import os
import argparse
from collections import OrderedDict
import random
from einops import rearrange, repeat
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import einsum, nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical
from local_attention import LocalAttention
from local_attention.rotary import apply_rotary_pos_emb


from utils import *


config = {
    'model_net': 'dt',  # mlp, cnn, cnn3d, transformer, transformercnn, dt
    'model': 'pg.episode.ppo.dt.roundly',
    'model_dir': 'models',
    'env_id': 'PongDeterministic-v0',
    'game_visible': False,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'wandb': 'pg.episode.ppo.dt.roundly',
    # 'run_in_notebook': True,
    'wandb': None,
    'run_in_notebook': False,

    'n_envs': 2,  # simultaneous processing environments
    'lr': 1e-4,
    'hidden_dim': 200,  # hidden size, linear units of the output layer
    'c_critic': 1.0,  # critic coefficient
    'c_entropy': 0.0,  # entropy coefficient
    'gamma': 0.99,
    'lam': 1.0,
    'surr_clip': 0.2,

    'transformer': {
        'num_blocks': 1,
        'dim_input': 128,  # dim_head * num_heads
        'dim_head': 16,
        'num_heads': 8,
        'state_dim': 80 * 80,
        'action_dim': 2,
        'reward_dim': 1,
        # 'window_size': 256,  # seq_len - 1
        'layer_norm': 'pre',  # pre/post
        'positional_encoding': 'none',  # relative/learned/rotary
        'dropout': 0.0,
    },

    'optimise_times': 1,  # optimise times per epoch
    'max_batch_size': 1000,  # mlp 10w, cnn 3w, cnn3d 3k, transformer 1k
    'chunk_percent': 1/16,  # split into chunks, do optimise per chunk
    'seq_len': 129,
    'epoch_episodes': 2,
    'epoch_save': 50,
    'max_epoch': 1000000,
    'target_reward': 20,
    'diff_state': False,
    'shuffle': False,
}
cfg = Config(**config)
cfg.transformer.window_size = cfg.seq_len - 1


class MyLocalAttention(LocalAttention):
    # key_padding_mask: (b, l), prefix padding, True -> keep
    def forward_simple(self, q, k, v, query_mask=None, key_mask=None, need_attn=False):
        q = q * (q.shape[-1] ** -0.5)
        if self.rel_pos is not None:
            pos_emb, xpos_scale = self.rel_pos(k)
            q, k = apply_rotary_pos_emb(q, k, pos_emb, scale = xpos_scale)

        sim = einsum('b h i e, b h j e -> b h i j', q, k)
        sim = self.masked_fill(sim, query_mask, key_mask)

        attn = sim.softmax(dim = -1)
        attn_weight = None
        if need_attn:
            attn_weight = attn.mean(dim=1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j e -> b h i e', attn, v)
        return out, attn_weight

    def masked_fill(self, sim, query_mask=None, key_mask=None):
        b, h, i, j = sim.shape
        # mask_value = -torch.finfo(sim.dtype).max
        mask_value = float("-inf")
        if key_mask is not None:
            mask = repeat(key_mask, 'b j -> b h i j', h=h, i=i)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        if query_mask is not None:
            mask = repeat(query_mask, 'b i -> b h i j', h=h, j=j)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        if self.causal or self.exact_windowsize:
            t_q = torch.arange(j, i + j).reshape(-1, 1)
            t_k = torch.arange(i, i + j)
            mask = torch.zeros(i, j, dtype=torch.bool)
            if self.causal:
                # 对齐右下角最后一个元素
                mask = mask | (t_q < t_k)
            if self.exact_windowsize:
                # mask element outside of sliding window
                mask = mask | (t_q > (t_k + self.window_size))
            mask = repeat(mask, 'i j -> b h i j', b=b, h=h)
            sim = sim.masked_fill(mask.to(sim.device), mask_value)
            del mask
        return sim


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim_head * cfg.num_heads

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)

        self.attn_fn = MyLocalAttention(
            dim = cfg.dim_head,
            window_size = cfg.window_size,
            causal = True,
            autopad = True,
            exact_windowsize = True,
            dropout = cfg.dropout,
            use_rotary_pos_emb = True if cfg.positional_encoding == 'rotary' else False
        )
        # self.attn_fn = nn.MultiheadAttention(inner_dim, cfg.num_heads, batch_first=True)

        if cfg.layer_norm == "pre":
            self.norm1 = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 4, dim)
        )


    def forward(self, x, mem_kv, query_mask=None, key_mask=None, mode='inference'):
        x_ = x

        if self.cfg.layer_norm == 'pre':
            x_ = self.norm1(x_)

        q = self.to_q(x_)
        kv = self.to_kv(x_)
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.cfg.num_heads), (q, kv))
        mem_kv = torch.cat([mem_kv, kv], dim=2)
        k, v = mem_kv.chunk(2, dim = -1)

        if mode == 'inference':
            h, attn_weight = self.attn_fn.forward_simple(q, k, v, query_mask=query_mask, key_mask=key_mask, need_attn=True)
        else:
            # h, attn_weight = self.attn_fn(q, k, v)
            h, attn_weight = self.attn_fn.forward_simple(q, k, v, need_attn=True)
        h = rearrange(h, 'b h n d -> b n (h d)')
        h = h + x

        h_ = h

        if self.cfg.layer_norm == 'pre':
            h_ = self.norm2(h)
        elif self.cfg.layer_norm == 'post':
            h = self.norm1(h)
            h_ = h

        out = self.to_out(h_)
        out = out + h

        if self.cfg.layer_norm == 'post':
            out = self.norm2(out)

        return out, attn_weight, mem_kv


class TransformerBaseModel(nn.Module):
    def __init__(self, cfg, num_outputs):
        super().__init__()
        self.cfg = cfg
        # memory is needed by inference
        self.memory = [Memory(cfg.window_size) for _ in range(cfg.num_blocks)]

        self.feature = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten(start_dim=2)),
            ('linear', nn.Linear(in_features=cfg.state_dim, out_features=cfg.dim_input)),
        ]))
        # prevent "dying ReLU" output zero
        nn.init.orthogonal_(self.feature.linear.weight, np.sqrt(2))

        assert cfg.num_blocks > 0
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg)
            for _ in range(cfg.num_blocks)])

    def clear_memory(self):
        for m in self.memory:
            m.clear()

    # mode: inference/train
    def forward(self, h, query_mask=None, key_mask=None, mode='inference'):
        attn_weights = []
        for i, block in enumerate(self.transformer_blocks):
            if mode == 'inference':
                mem_kv = self.memory[i].get()
                mask_size = h.shape[-2] if self.memory[i].is_empty() else (mem_kv.shape[-2] + h.shape[-2])
                h, attn_weight, mem_kv = block(h, mem_kv, query_mask, key_mask[:, -mask_size:], mode)
                self.memory[i].set(mem_kv.detach())
            else:
                h, attn_weight, _ = block(h, Memory.get_empty_mem(), mode=mode)
            attn_weights.append(attn_weight)
        return h, attn_weights


class TransformerModel(TransformerBaseModel):
    def __init__(self, cfg, num_outputs):
        super().__init__(cfg, num_outputs)
        self.actor = nn.Sequential(
            nn.Linear(in_features=cfg.dim_input, out_features=num_outputs),
        )
        self.critic = nn.Sequential(
            nn.Linear(in_features=cfg.dim_input, out_features=1),
        )

    def forward(self, x, query_mask=None, key_mask=None, mode='inference', lookback=0):
        h = self.feature(x)
        h, attn_weights = super().forward(h, query_mask, key_mask, mode)
        if mode == 'inference':
            h = h.squeeze(1)
        else:
            h = h.squeeze(0)
            h = h[lookback:, ...]
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value, attn_weights


class SeqConv2d(nn.Conv2d):
    def forward(self, x):
        n, l, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = super().forward(x)
        x = x.view(n, l, *x.shape[1:])
        return x

        
class TransformerCNNModel(TransformerModel):
    def __init__(self, cfg, num_outputs):
        super().__init__(cfg, num_outputs)
        self.feature = nn.Sequential(OrderedDict([
            ('conv1', SeqConv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4, padding=2)),
            ('act1', nn.ReLU()),
            ('conv2', SeqConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('act2', nn.ReLU()),
            ('flatten', nn.Flatten(start_dim=2)),
            ('linear', nn.Linear(in_features=32 * 10 * 10, out_features=cfg.dim_input)),
        ]))
        # prevent "dying ReLU" output zero
        nn.init.orthogonal_(self.feature.linear.weight, np.sqrt(2))


class DTModel(TransformerCNNModel):
    def __init__(self, cfg, num_outputs):
        super().__init__(cfg, num_outputs)
        # self.cfg = cfg.transformer
        cfg = self.cfg

        self.embed_state = self.feature
        # self.embed_reward = nn.Embedding(cfg.reward_dim, cfg.dim_input)
        self.embed_action = nn.Embedding(cfg.action_dim, cfg.dim_input)
        self.embed_reward = nn.Linear(cfg.reward_dim, cfg.dim_input)
        # self.embed_action = nn.Linear(num_outputs, cfg.dim_input)
        self.embed_timestep = nn.Embedding(20000, cfg.dim_input)

        self.predict_state = nn.Linear(cfg.dim_input, cfg.state_dim)
        self.predict_action = nn.Linear(cfg.dim_input, num_outputs)
        self.predict_reward = nn.Linear(cfg.dim_input, 1)

    # mode: inference/train
    def forward(self, rewards, states, actions, timesteps, query_mask=None, key_mask=None, mode='inference', lookback=0):
        reward_embeddings = self.embed_reward(rewards)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)

        reward_embeddings = reward_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack([reward_embeddings, state_embeddings, action_embeddings], dim=2)
        stacked_inputs = rearrange(stacked_inputs, 'b n m d -> b (n m) d')

        stacked_query_mask, stacked_key_mask = None, None
        if query_mask is not None:
            stacked_query_mask = repeat(query_mask, 'b i -> b (i m)', m=3)
        if key_mask is not None:
            stacked_key_mask = repeat(key_mask, 'b j -> b (j m)', m=3)

        h, attn_weights = TransformerBaseModel.forward(self, stacked_inputs, stacked_query_mask, stacked_key_mask, mode)

        if mode == 'inference':
            next_actions = self.predict_action(h[:, 1::3].squeeze(1))
            next_rewards = None
            next_states = None
        else:
            h = h.squeeze(0)
            h = h[3 * lookback:, ...]
            next_actions = self.predict_action(h[1::3])
            next_rewards = self.predict_reward(h[2::3])
            next_states = self.predict_state(h[2::3])
        return next_states, next_actions, next_rewards, attn_weights


class Memory:
    def __init__(self, size):
        self.size = size
        self.mem = self.get_empty_mem()

    @staticmethod
    def get_empty_mem():
        return torch.tensor([]).to(cfg.device)

    def get(self):
        return self.mem

    def set(self, mem):
        self.mem = mem[..., -self.size:, :]

    def clear(self):
        self.mem = self.get_empty_mem()

    def is_empty(self):
        return self.mem.shape[0] == 0

    # def add(self, data):
    #     if self.mem.shape[0] == 0:
    #         self.mem = data.clone()
    #     else:
    #         self.mem = torch.cat([self.mem, data], dim=1)  # along sequence dim
    #         self.mem = self.mem[..., -self.size:, :]


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
        x = x.permute(0, 2, 1, 3, 4)
        feature = self.feature(x)
        logits = self.actor(feature)
        dist = Categorical(logits=logits)
        value = self.critic(feature)
        return dist, value


class ConcatenateModule(nn.Module):
    def __init__(self, dim=-1):
        super(ConcatenateModule, self).__init__()
        self.dim = dim

    def forward(self, tensors):
        # Concatenate the two input tensors along the specified dimension
        return torch.cat(tensors, dim=self.dim)


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
        self.feature = nn.Sequential(OrderedDict([
            ('concat', ConcatenateModule(dim=-1)),
            ('linear', nn.Linear(in_features=cfg.hidden_dim * 2, out_features=cfg.hidden_dim)),
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
        x = x.permute(0, 2, 1, 3, 4)
        feature3d = self.feature3d(x)
        feature2d = self.feature2d(x[:, :, -1, ...])
        # feature = torch.cat([feature3d, feature2d], dim=-1)
        feature = self.feature([feature3d, feature2d])
        logits = self.actor(feature)
        dist = Categorical(logits=logits)
        value = self.critic(feature)
        return dist, value


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

    def pending_len(self):
        return len(self.actions)


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

    def peek_batch(self, index, seq_len, padding=0):
        new_data = []
        for i in range(self.n):
            pack = self.episode_collectors[i].peek(seq_len)
            seq_data = pack[index]
            d = seq_len - len(seq_data)
            if d > 0:
                seq_data = [padding] * d + seq_data
            assert len(seq_data) == seq_len
            new_data.append(seq_data)
        return np.array(new_data)

    def peek_and_append(self, index, seq_len, elem):
        seq = self.peek_batch(index, seq_len - 1, elem[0] - elem[0])
        seq = np.concatenate([seq, elem[:, np.newaxis]], 1)
        assert seq.shape[1] == seq_len
        return seq

    def peek_state(self, seq_len, state):
        return self.peek_and_append(0, seq_len, state)
        # seq_state = self.peek_batch(0, seq_len - 1, state[0] - state[0])
        # seq_state = np.concatenate([seq_state, state[:, np.newaxis, ...]], 1)
        # assert seq_state.shape[1] == seq_len
        # return seq_state

    def peek_mask(self, seq_len):
        # some episode may not collect enough data
        seq_done = self.peek_batch(3, seq_len - 1, True)
        b, l = seq_done.shape
        done = np.array([False] * b).reshape([b, 1])
        seq_done = np.concatenate([seq_done, done], 1)
        assert seq_done.shape[1] == seq_len

        mask = 1 - seq_done
        mask = mask[:, ::-1]
        mask = np.minimum.accumulate(mask, 1)
        mask = mask[:, ::-1]
        return mask.copy()  # return copy to prevent reverse index error when converting torch tensor

    def peek_timestep(self):
        timestep = []
        for i in range(self.n):
            timestep.append(self.episode_collectors[i].pending_len())
        return timestep


class BaseGenerator:
    # (s1, s2, ...), (a1, a2, ...), (r1, r2, ...), ...
    def __init__(self, batch_size, data_fn, random=False, *data):
        self.batch_size = batch_size
        self.data_fn = data_fn
        self.random = random
        self.data = list(data)
        self.iter = 0

    def __len__(self):
        return len(self.data[0]) if len(self.data) > 0 else 0

    def shuffle(self):
        indices = list(range(len(self)))
        random.shuffle(indices)
        for i, d in enumerate(self.data):
            self.data[i] = [d[i] for i in indices]

    def normalize(self, index):
        d = np.array(self.data[index])
        d = normalize(d)
        self.data[index] = list(d)

    def inc_iter(self):
        self.iter += 1

    def get_iter(self):
        return self.iter

    def next_batch(self):
        if self.random:
            self.shuffle()
        for i in range(0, len(self), self.batch_size):
            yield self.data_fn(*[d[i:i + self.batch_size] for d in self.data])
            self.inc_iter()


class EpDataGenerator(BaseGenerator):
    def __init__(self, episodes, batch_size, data_fn, random=False):
        states, actions, rewards, dones = [], [], [], []
        extra = []
        round_count = 0
        for episode in episodes:
            # ep_rewards is original rewards, discount rewards are in extra
            ep_states, ep_actions, ep_rewards, ep_dones, *ep_extra = episode
            round_count += (np.array(ep_rewards) != 0).sum().item()
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


class LookBackEpDataGenerator(EpDataGenerator):
    def __init__(self, episodes, batch_size, lookback, data_fn, offset=0):
        super().__init__(episodes, batch_size, data_fn, False)
        self.lookback = lookback
        self.offset = offset

    def get_dynamic_offset(self):
        return self.offset

    def next_batch(self):
        for i in range(self.offset, len(self), self.batch_size):
            self.offset = min(i, self.lookback)
            yield self.data_fn(*[d[i - self.offset:i + self.batch_size] for d in self.data])
            self.inc_iter()


class LookBackSeqDataGenerator(BaseGenerator):
    """_summary_

    Args:
        lookback: window size
        offset: where first data starting from
    """
    def __init__(self, batch_size, lookback, offset, data_fn, *data):
        super().__init__(batch_size, data_fn, False, *data)
        self.lookback = lookback
        self.offset = offset

    def get_dynamic_offset(self):
        return self.offset

    def next_batch(self):
        dones = self.data[3]
        total = len(self)
        last_done, curr_done = -1, -1
        for i in range(self.offset):
            if dones[i]:
                last_done = i
        i, start = self.offset, self.offset
        end = min(start + self.batch_size, total)
        while True:
            if i == end:
                # look back window size
                lookback_pos = max(0, start - self.lookback)
                lookback_pos = max(lookback_pos, last_done + 1)
                # strip size
                self.offset = start - lookback_pos
                # only lookback for states
                # yield self.data_fn(*[d[lookback_pos:end] if j == 0 else d[start:end] for j, d in enumerate(self.data)])
                yield self.data_fn(*[d[lookback_pos:end] for d in self.data])
                self.inc_iter()

                if i >= total:
                    assert i == total
                    break

                start = end
                end = min(start + self.batch_size, total)
                if curr_done != -1:
                    last_done = curr_done
                curr_done = -1
            if dones[i]:
                end = i + 1
                curr_done = i
            i += 1


def discount_rewards_roundly(rewards, gamma=1.0):
    # discounted_rewards = [0] * len(rewards)
    discounted_rewards = np.zeros_like(rewards)
    cum_reward = 0
    for i in reversed(range(len(rewards))):
        if rewards[i] != 0:
            cum_reward = 0  # reset the sum, since this was a game boundary (pong specific!)
        cum_reward = rewards[i] + gamma * cum_reward
        discounted_rewards[i] = cum_reward
    return discounted_rewards


def discount_rewards_episodely(rewards, gamma=1.0):
    discounted_rewards = np.zeros_like(rewards)
    cum_reward = 0
    for i in reversed(range(len(rewards))):
        # if rewards[i] != 0:
        #     cum_reward = 0
        cum_reward = rewards[i] + gamma * cum_reward
        discounted_rewards[i] = cum_reward
    return discounted_rewards


def discount_gae_roundly(rewards, values, gamma=0.99, lam=0.95):
    returns = np.zeros_like(rewards)
    gae = 0
    next_value = 0
    for i in reversed(range(len(rewards))):
        if rewards[i] != 0:
            gae = 0
            next_value = 0
        delta = rewards[i] + gamma * next_value - values[i]
        gae = delta + lam * gamma * gae
        returns[i] = values[i] + gae
        next_value = values[i]
    return returns


# def discount_gae(rewards, values, gamma=0.99, lam=0.95):
#     advantages = np.zeros_like(rewards)
#     cum_gae = 0
#     next_value = 0
#     for i in reversed(range(len(rewards))):
#         if rewards[i] != 0:
#             cum_gae = 0
#         delta = rewards[i] + gamma * next_value - values[i]
#         cum_gae = delta + lam * gamma * cum_gae
#         next_value = values[i]
#         advantages[i] = cum_gae
#     return advantages


# def discount_gae2(rewards, values, gamma=0.99, lam=0.95):
#     advantages = np.zeros_like(rewards)
#     cum_gae = 0
#     next_value = 0
#     for i in reversed(range(len(rewards))):
#         if rewards[i] != 0:
#             cum_gae = 0
#             next_value = 0
#         delta = rewards[i] + gamma * next_value - values[i]
#         cum_gae = delta + lam * gamma * cum_gae
#         next_value = values[i]
#         advantages[i] = cum_gae
#     return advantages


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
    return {'critic_loss': critic_loss, 'actor_loss': actor_loss, 'entropy_loss': entropy_loss, 'loss': loss}


def loss_ppo(trainer, data_loader, states, actions, rewards, old_log_probs, advantages):
    if cfg.model_net in ('transformer', 'transformercnn'):
        lookback = data_loader.get_dynamic_offset()
        dist, values, attn_weights = trainer.model(states.unsqueeze(0), mode='train', lookback=lookback)
        states, actions, rewards, old_log_probs, advantages = states[lookback:], actions[lookback:], rewards[lookback:], old_log_probs[lookback:], advantages[lookback:]
        if data_loader.get_iter() == 0:
            trainer.writer.summary_attns([attn[0].unsqueeze(0) for attn in attn_weights])
    else:
        dist, values = trainer.model(states)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    values = values.squeeze(1)

    # todo: normalize?
    advantages = rewards - values.detach()
    # no norm for transformer, will cause NaN
    # advantages = normalize(advantages)
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - cfg.surr_clip, 1.0 + cfg.surr_clip) * advantages
    actor_loss = -torch.min(surr1, surr2).sum() / len(data_loader)

    critic_loss = 0.5 * (values - rewards).pow(2).sum() / len(data_loader)
    entropy_loss = -entropy.sum() / len(data_loader)
    loss = actor_loss + critic_loss * cfg.c_critic + entropy_loss * cfg.c_entropy
    return {'critic_loss': critic_loss, 'actor_loss': actor_loss, 'entropy_loss': entropy_loss, 'loss': loss}


class Train:
    def __init__(self):
        self.writer = MySummaryWriter(0, cfg.epoch_save // 4, comment=f'.{cfg.model}.{cfg.env_id}')
        self.cum_reward = None

    @staticmethod
    def diff_state(state, last_state):
        return state if not cfg.diff_state else (state - last_state)

    @staticmethod
    def make_envs():
        def _init(env_id, seed=None, **kwargs):
            env = gym.make(env_id, **kwargs)
            env.seed(seed)  # Apply a unique seed to each environment
            return env
        envs = [lambda: _init(cfg.env_id, seed=np.random.randint(0, 10000), render_mode='human' if cfg.game_visible else 'rgb_array')] * cfg.n_envs
        envs = SubprocVecEnv(envs)
        return envs

    @staticmethod
    def get_model(cfg, num_outputs=0):
        if cfg.model_net == 'mlp':
            return MLPModel(cfg, num_outputs)
        elif cfg.model_net == 'cnn':
            return CNNModel(cfg, num_outputs)
        elif cfg.model_net == 'cnn3d':
            return CNN3dModel(cfg, num_outputs)
        elif cfg.model_net == 'cnn3d2d':
            return CNN3d2dModel(cfg, num_outputs)
        elif cfg.model_net == 'transformer':
            return TransformerModel(cfg.transformer, num_outputs)
        elif cfg.model_net == 'transformercnn':
            return TransformerCNNModel(cfg.transformer, num_outputs)
        elif cfg.model_net == 'dt':
            return DTModel(cfg.transformer, num_outputs)

    @staticmethod
    def load_model(load_from, model, optimizer):
        epoch = 0
        if load_from is not None:
            checkpoint = torch.load(load_from, map_location=cfg.device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            print(f'Model loaded, starting from epoch {epoch}')
        return epoch

    @staticmethod
    def save_model(model, optimizer, epoch, avg_reward):
        name = "%s_%s_%+.3f_%d.pth" % (cfg.model, cfg.env_id, avg_reward, epoch)
        fname = os.path.join(cfg.model_dir, cfg.model, name)
        states = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(states, fname)

    def data_fn(self, states, actions, rewards, *extra):
        states = torch.as_tensor(np.array(states), dtype=torch.float32).to(cfg.device)  # requires_grad=False
        actions = torch.as_tensor(actions).to(cfg.device)  # requires_grad=False
        rewards = torch.as_tensor(rewards).to(cfg.device)  # requires_grad=False
        extra = [torch.as_tensor(e).to(cfg.device) for e in extra]
        return states, actions, rewards, *extra

    def get_chunk_loader(self, episodes, chunk_size, data_fn, random=False):
        return EpDataGenerator(episodes, chunk_size, data_fn, random=random)

    def get_minibatch_loader(self, chunk_loader, max_batch_size, data_fn, random, *chunk):
        return BaseGenerator(max_batch_size, data_fn, random, *chunk)

    def begin_epoch_optimize(self, epoch):
        self.writer.update_global_step(epoch)

    def end_epoch_optimize(self, epoch):
        self.collector.clear_all()

    def get_model_params(self, collector, state):
        # return tuple
        return torch.FloatTensor(state).to(cfg.device),

    def predict_action(self, dist, value):
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach(), value.detach()

    # episodes: [(states, actions, rewards, ...), ...]
    def process_episodes(self, episodes):
        avg_reward = 0
        total_samples = 0
        cum_reward = self.cum_reward
        epoch = self.writer.global_step
        for i, episode in enumerate(episodes):
            ep_states, ep_actions, ep_rewards, ep_dones, *_ = episode
            ep_reward_sum, ep_steps = sum(ep_rewards), len(ep_rewards)
            avg_reward += ep_reward_sum
            cum_reward = ep_reward_sum if cum_reward is None else cum_reward * 0.99 + ep_reward_sum * 0.01
            print(f'Run {(epoch - 1) * cfg.epoch_episodes + i + 1}, steps {ep_steps}, reward {ep_reward_sum}, cum_reward {cum_reward:.3f}')
            total_samples += len(ep_rewards)

            self.recalc_episode(episode)

        avg_reward /= len(episodes)
        self.writer.add_scalar('Reward/epoch_reward_avg', avg_reward, self.writer.global_step)
        self.writer.add_scalar('Reward/env_1_reward', sum(episodes[0][2]), self.writer.global_step)
        self.cum_reward = cum_reward
        return total_samples, avg_reward

    def recalc_episode(self, episode):
        ep_states, ep_actions, ep_rewards, ep_dones, ep_log_probs, ep_values = episode
        ep_rewards = np.array(ep_rewards)
        ep_discount_rewards = discount_rewards_roundly(ep_rewards, cfg.gamma)
        # ep_discount_rewards = discount_gae_roundly(ep_rewards, ep_values, cfg.gamma, cfg.lam)
        # ep_rewards = normalize(ep_rewards)
        # ep_rewards = list(ep_rewards)
        # episode[2] = ep_rewards
        episode.insert(4, list(ep_discount_rewards))
        ep_values = np.array([v.item() for v in ep_values])
        ep_advantages = ep_discount_rewards - ep_values
        episode[-1] = list(ep_advantages)  # replace ep_values with ep_advantages

    def loss(self, data_loader, states, actions, rewards, old_log_probs, advantages):
        return loss_ppo(self, data_loader, states, actions, rewards, old_log_probs, advantages)

    def optimise_by_minibatch(self, model, optimizer, data_loader):
        acc_loss = {}
        optimizer.zero_grad()
        for states, actions, rewards, dones, discount_rewards, *extra in data_loader.next_batch():
            loss_dict = self.loss(data_loader, states, actions, discount_rewards, *extra)

            # summary in the first iteration and zero_grad after that
            if data_loader.get_iter() == 0:
                params_feature = model.get_parameter('feature.linear.weight')
                self.writer.summary_grad(optimizer, params_feature, loss_dict)
                optimizer.zero_grad()

            loss_dict['loss'].backward()

            keys = set(acc_loss).union(loss_dict)
            acc_loss = {key: acc_loss.get(key, 0) + loss_dict.get(key, 0) for key in keys}
        optimizer.step()
        self.writer.summary_loss(acc_loss)

    def train(self, load_from):
        writer = self.writer
        if not cfg.run_in_notebook:
            writer.summary_script_content(__file__)
        else:
            writer.summary_text('', f'```python\n{In[-1]}\n```')

        envs = self.make_envs()
        num_outputs = 2  #envs.action_space.n

        model = self.model = self.get_model(cfg, num_outputs).to(cfg.device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)  # implements Adam algorithm
        epoch = self.load_model(load_from, model, optimizer)
        print(cfg)
        print(model)
        print(optimizer)

        collector = self.collector = BatchEpisodeCollector(cfg.n_envs)
        state = envs.reset()
        state = batch_prepro(state)
        last_state = state.copy()

        early_stop = False

        while not early_stop and epoch < cfg.max_epoch:
            state_ = self.diff_state(state, last_state).transpose(0, 3, 1, 2)

            with torch.no_grad():
                params = self.get_model_params(collector, state_)
                action, *extra = self.predict_action(*model(*params))

            action = action.cpu().numpy()
            next_state, reward, done, _ = envs.step(action + 2)
            next_state = batch_prepro(next_state)

            collector.add(state_, action, reward, done, *extra)
            last_state = state
            state = next_state
            # reset last_state to state when done
            for i, d in enumerate(done):
                if d:
                    last_state[i] = state[i]

            if collector.has_full_batch(cfg.epoch_episodes):
                epoch += 1
                self.begin_epoch_optimize(epoch)

                episodes = collector.roll_batch()
                total_samples, avg_reward = self.process_episodes(episodes)

                for _ in range(cfg.optimise_times):
                    # optimise every chunk_percent samples to accelerate training
                    chunk_size = int(np.ceil(total_samples * cfg.chunk_percent))
                    chunk_loader = self.get_chunk_loader(episodes, chunk_size, lambda *d: d, cfg.shuffle)

                    for chunk in chunk_loader.next_batch():
                        # minibatch don't exceeds cfg.max_batch_size
                        minibatch_loader = self.get_minibatch_loader(chunk_loader, cfg.max_batch_size, self.data_fn, False, *chunk)
                        self.optimise_by_minibatch(model, optimizer, minibatch_loader)
                # write summary once after optimise_times to prevent duplicate
                writer.write_summary()

                state = envs.reset()
                state = batch_prepro(state)
                last_state = state.copy()
                self.end_epoch_optimize(epoch)

                if epoch % cfg.epoch_save == 0:
                    self.save_model(model, optimizer, epoch, avg_reward)

                if avg_reward > cfg.target_reward:
                    early_stop = True


class TrainCNN3d(Train):
    def get_model_params(self, collector, state):
        state_ = collector.peek_state(cfg.seq_len, state)
        # return tuple
        return torch.FloatTensor(state_).to(cfg.device),

    def get_chunk_loader(self, episodes, chunk_size, data_fn, random=False):
        chunk_loader = StateSeqEpDataGenerator(episodes, chunk_size, cfg.seq_len, data_fn, random=random)
        # chunk_loader.normalize(-1)  # advantages index: -1
        return chunk_loader


class TrainTransformer(Train):
    def get_model_params(self, collector, state):
        state_ = state
        mask = collector.peek_mask(cfg.seq_len)
        pt_state = torch.as_tensor(state_, dtype=torch.float32).unsqueeze(1).to(cfg.device)
        pt_query_mask = None
        pt_key_mask = torch.as_tensor(mask, dtype=torch.bool).to(cfg.device)
        return pt_state, pt_query_mask, pt_key_mask

    def predict_action(self, dist, value, attn_weights):
        return super().predict_action(dist, value)

    def get_chunk_loader(self, episodes, chunk_size, data_fn, random=False):
        return LookBackEpDataGenerator(episodes, chunk_size, cfg.transformer.window_size, data_fn)

    def get_minibatch_loader(self, chunk_loader, max_batch_size, data_fn, random, *chunk):
        return LookBackSeqDataGenerator(max_batch_size, cfg.transformer.window_size, chunk_loader.get_dynamic_offset(), data_fn, *chunk)

    def begin_epoch_optimize(self, epoch):
        super().begin_epoch_optimize(epoch)
        self.model.clear_memory()

    def end_epoch_optimize(self, epoch):
        super().end_epoch_optimize(epoch)
        self.model.clear_memory()


class TrainDT(TrainTransformer):
    def get_model_params(self, collector: BatchEpisodeCollector, state):
        b = state.shape[0]
        state_ = state
        # todo
        action = np.zeros([b, ], dtype=np.int64)
        reward = np.ones([b, 1])
        timestep = np.array(collector.peek_timestep())
        mask = collector.peek_mask(cfg.seq_len)
        pt_state = torch.as_tensor(state_, dtype=torch.float32).unsqueeze(1).to(cfg.device)
        pt_action = torch.as_tensor(action, dtype=torch.int64).unsqueeze(1).to(cfg.device)
        pt_reward = torch.as_tensor(reward, dtype=torch.float32).unsqueeze(1).to(cfg.device)
        pt_timestep = torch.as_tensor(timestep).unsqueeze(1).to(cfg.device)
        pt_key_mask = torch.as_tensor(mask, dtype=torch.bool).to(cfg.device)
        pt_query_mask = None
        # pt_query_mask = torch.ones(state.shape[0], 3, dtype=torch.bool)
        # pt_query_mask[:, -1] = False
        # pt_query_mask = pt_query_mask.to(cfg.device)
        return pt_reward, pt_state, pt_action, pt_timestep, pt_query_mask, pt_key_mask

    def predict_action(self, states, action_logits, rewards, attn_weights):
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        # return tuple
        return action,

    def recalc_episode(self, episode):
        ep_states, ep_actions, ep_rewards, ep_dones = episode
        ep_rewards = np.array(ep_rewards)
        # no discount
        ep_discount_rewards = discount_rewards_roundly(ep_rewards, 1.0)
        episode.insert(4, list(ep_discount_rewards))
        episode.insert(5, list(range(len(ep_rewards))))

    def data_fn(self, states, actions, rewards, dones, discount_rewards, timesteps):
        states = torch.as_tensor(np.array(states), dtype=torch.float32).to(cfg.device)  # requires_grad=False
        actions = torch.as_tensor(actions, dtype=torch.int64).to(cfg.device)  # requires_grad=False
        rewards = torch.as_tensor(rewards).to(cfg.device)  # requires_grad=False
        dones = torch.as_tensor(dones).to(cfg.device)  # requires_grad=False
        discount_rewards = torch.as_tensor(discount_rewards, dtype=torch.float32).to(cfg.device)  # requires_grad=False
        timesteps = torch.as_tensor(timesteps).to(cfg.device)  # requires_grad=False
        return states, actions, rewards, dones, discount_rewards, timesteps

    def loss(self, data_loader, states, actions, rewards, timesteps):
        lookback = data_loader.get_dynamic_offset()
        # lookforward = data_loader.get_dynamic_offset()
        next_states, next_action_logits, next_rewards, attn_weights = self.model(
            rewards.reshape([1, -1, 1]), states.unsqueeze(0), actions.reshape([1, -1]), timesteps.reshape([1, -1]),
            mode='train', lookback=lookback)

        if data_loader.get_iter() == 0:
            self.writer.summary_attns([attn[0].unsqueeze(0) for attn in attn_weights])

        action_loss = F.cross_entropy(next_action_logits, actions[lookback:], reduction='sum') / len(data_loader)

        dist = Categorical(logits=next_action_logits)
        entropy = dist.entropy()
        entropy_loss = -entropy.sum() / len(data_loader)

        loss = action_loss + entropy_loss * cfg.c_entropy
        return {'action_loss': action_loss, 'entropy_loss': entropy_loss, 'loss': loss}


def test_env(env, model):
    state, _ = env.reset()
    state = prepro(state)
    last_state = state.copy()

    collector = BatchEpisodeCollector(1)
    done = False
    total_reward = 0
    steps = 0
    while not done:
        state_ = diff_state(state, last_state).transpose(2, 0, 1)
        if cfg.model_net in ('cnn3d', 'cnn3d2d'):
            state_ = collector.peek_state(cfg.seq_len, state_[np.newaxis, ...])
        if cfg.model_net in ('transformer', 'transformercnn'):
            mask = collector.peek_mask(cfg.seq_len)
            dist, value = model(torch.as_tensor(state_, dtype=torch.float32).unsqueeze(1).to(cfg.device),
                                torch.as_tensor(mask, dtype=torch.bool).to(cfg.device))
        else:
            state_ = torch.FloatTensor(state_).to(cfg.device)
            dist, value = model(state_)
        # dist = model(torch.FloatTensor(diff_state(state, last_state)).unsqueeze(0).to(cfg.device))
        action = dist.sample()
        action = action.cpu().numpy()[0]
        next_state, reward, done, _, _ = env.step(action + 2)
        next_state = prepro(next_state)

        collector.add([diff_state(state, last_state).transpose(2, 0, 1)], [action], [reward], [done])
        last_state = state
        state = next_state
        total_reward += reward
        steps += 1
    collector.clear_all()
    return total_reward, steps


def train(load_from=None):
    wandb_init(cfg.wandb, cfg)
    if cfg.model_net in ('cnn3d', 'cnn3d2d'):
        TrainCNN3d().train(load_from)
    elif cfg.model_net in ('transformer', 'transformercnn'):
        TrainTransformer().train(load_from)
    elif cfg.model_net == 'dt':
        TrainDT().train(load_from)
    else:
        Train().train(load_from)
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


def test_StateSeqEpDataGenerator():
    episode = [
        [i+1 for i in range(10)],
        [i+1 for i in range(10)],
        [False, False, True, False, False, True, False, False, False, True],
        [False, False, True, False, False, True, False, False, False, True],
    ]
    g = StateSeqEpDataGenerator([episode] * 2, 2, 3, data_fn)
    assert g.data[0] == [[0, 0, 1], [0, 1, 2], [1, 2, 3], [0, 0, 4], [0, 4, 5], [4, 5, 6], [0, 0, 7], [0, 7, 8], [7, 8, 9], [8, 9, 10], [0, 0, 1], [0, 1, 2], [1, 2, 3], [0, 0, 4], [0, 4, 5], [4, 5, 6], [0, 0, 7], [0, 7, 8], [7, 8, 9], [8, 9, 10]]


def test_LookBackSeqDataGenerator():
    data = [[False, False, False, False, True, False, False, True, False, True]] * 4
    g = LookBackSeqDataGenerator(2, 2, 0, data_fn, *data)
    iter = g.next_batch()
    v = next(iter)
    assert v[0].tolist() == [False, False]
    v = next(iter)
    assert v[0].tolist() == [False, False, False, False]
    v = next(iter)
    assert v[0].tolist() == [False, False, True]
    v = next(iter)
    assert v[0].tolist() == [False, False]
    v = next(iter)
    assert v[0].tolist() == [False, False, True]
    v = next(iter)
    assert v[0].tolist() == [False, True]
    
    data = [[False, False, False, False, False, False, False, False, False, False]] * 4
    g = LookBackSeqDataGenerator(6, 3, 0, data_fn, *data)
    iter = g.next_batch()
    _, v = next(iter), next(iter)
    assert v[0].tolist() == [False, False, False, False, False, False, False]

    data = [[True, True, False, False, False, False, False, False, True]] * 4
    g = LookBackSeqDataGenerator(2, 3, 0, data_fn, *data)
    iter = g.next_batch()
    v = next(iter)
    assert v[0].tolist() == [True]
    v = next(iter)
    assert v[0].tolist() == [True]
    _, v = next(iter), next(iter)
    assert v[0].tolist() == [False, False, False, False]
    v = next(iter)
    assert v[0].tolist() == [False, False, False, False, False]
    v = next(iter)
    assert v[0].tolist() == [False, False, False, True]

    data = [[True, False, False, False, False, False, False, False, False]] * 4
    g = LookBackSeqDataGenerator(4, 3, 2, data_fn, *data)
    iter = g.next_batch()
    v = next(iter)
    assert v[0].tolist() == [False, False, False, False, False]
    v = next(iter)
    assert v[0].tolist() == [False, False, False, False, False, False]

    data = [[False, False, False, False, False, False, False, False, False]] * 4
    g = LookBackSeqDataGenerator(4, 3, 3, data_fn, *data)
    iter = g.next_batch()
    v = next(iter)
    assert v[0].tolist() == [False, False, False, False, False, False, False]
    v = next(iter)
    assert v[0].tolist() == [False, False, False, False, False]


def test_MyLocalAttention():
    attn = MyLocalAttention(
            dim = cfg.transformer.dim_head,
            window_size = cfg.transformer.window_size,
            causal = True,
            autopad = True,
            exact_windowsize = True,
        )
    b, h, i, j = 2, 3, 1, 5
    sim = torch.ones(b, h, i, j, dtype=torch.float32)
    key_padding_mask = torch.ones(b, j, dtype=torch.bool)
    sim_masked = attn.masked_fill(sim, key_padding_mask)
    assert (sim_masked == sim).all()

    key_padding_mask = torch.tensor([[False, False, False, False, True],
                                     [False, False, False, False, True]], dtype=torch.bool)
    sim_masked = attn.masked_fill(sim, key_padding_mask)
    sim_target = sim.clone()
    sim_target[:, :, :, 0:4] = float("-inf")
    assert (sim_masked == sim_target).all()

    key_padding_mask = torch.tensor([[False, False, True, True, True],
                                     [False, True, True, True, True]], dtype=torch.bool)
    sim_masked = attn.masked_fill(sim, key_padding_mask)
    sim_target = sim.clone()
    sim_target[0, :, :, 0:2] = float("-inf")
    sim_target[1, :, :, 0:1] = float("-inf")
    assert (sim_masked == sim_target).all()

    b, h, i, j = 2, 3, 5, 5
    sim = torch.ones(b, h, i, j, dtype=torch.float32)
    sim_masked = attn.masked_fill(sim)
    sim_target = sim.clone()
    sim_target[:, :, 0, 1:] = float("-inf")
    sim_target[:, :, 1, 2:] = float("-inf")
    sim_target[:, :, 2, 3:] = float("-inf")
    sim_target[:, :, 3, 4:] = float("-inf")
    assert (sim_masked == sim_target).all()


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
