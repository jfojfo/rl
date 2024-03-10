from collections import OrderedDict
import torch
from torch import einsum, nn
from torch.distributions import Categorical
from einops import rearrange, repeat
from local_attention import LocalAttention
from local_attention.rotary import apply_rotary_pos_emb
from utils import Config


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
            ('act', nn.ReLU()),
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
            ('act3', nn.ReLU()),
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
            ('act3', nn.ReLU()),
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
            ('act3', nn.ReLU()),
        ]))
        self.feature2d = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4, padding=2)),
            ('act1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)),
            ('act2', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear', nn.Linear(in_features=32 * 10 * 10, out_features=cfg.hidden_dim)),
            ('act3', nn.ReLU()),
        ]))
        self.feature = nn.Sequential(OrderedDict([
            ('concat', ConcatenateModule(dim=-1)),
            ('linear', nn.Linear(in_features=cfg.hidden_dim * 2, out_features=cfg.hidden_dim)),
            ('act', nn.ReLU()),
        ]))
        self.actor = nn.Sequential(
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
