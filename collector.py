import random
import numpy as np
from utils import normalize


class Collector:
    StateIndex = 0
    ActionIndex = 1
    RewardIndex = 2
    DoneIndex = 3


class SeqStepCollector:
    def __init__(self, seq_len, lookback=0):
        self.seq_len = seq_len
        self.lookback = lookback
        self.seq = []
        self.count = 0

    def add(self, item):
        self.seq.append(item)
        pos = self.seq_len + self.lookback
        self.seq = self.seq[-pos:]
        self.count += 1

    def steps_count(self):
        return self.count

    def roll(self):
        ret = self.seq
        self.seq = ret[-self.lookback:]
        self.count = 0
        return ret

    def peek(self, n):
        pos = n + self.lookback
        return self.seq[-pos:]


class MultiSeqStepCollector(Collector):
    def __init__(self, seq_len, lookback=0, lookforward=0):
        self.seq_len = seq_len
        self.lookback = lookback
        self.lookforward = lookforward
        self.seq_list = None

    # state, action, reward, done, next_state, *extra_pt_tensor
    def add(self, state, action, reward, done, next_state, *extra):
        # prevent torch.as_tensor throwing warning for np.bool_ array
        items = (state, action, reward, done.astype(np.int32)) + extra
        self.next_state = next_state
        if self.seq_list is None:
            self.seq_list = [SeqStepCollector(self.seq_len + self.lookback) for _ in range(len(items))]
        for i, item in enumerate(items):
            self.seq_list[i].add(item)

    def has_full_batch(self, n_steps):
        if self.seq_list is None or len(self.seq_list) == 0:
            return False
        curr_steps = self.seq_list[0].steps_count()
        assert curr_steps <= n_steps
        return curr_steps == n_steps

    def roll_batch(self):
        return [seq.roll() for seq in self.seq_list]

    def roll_batch_with_index(self):
        return {i: seq.roll() for i, seq in enumerate(self.seq_list)}

    def clear_all(self):
        pass

    def peek_batch(self, index, seq_len, padding=0):
        assert index < len(self.seq_list)
        seq_data = self.seq_list[index].peek()
        d = seq_len - len(seq_data)
        if d > 0:
            seq_data = [padding] * d + seq_data
        assert len(seq_data) == seq_len
        return seq_data

    def peek_and_append(self, index, seq_len, elem, padding):
        seq = self.peek_batch(index, seq_len - 1, padding)
        seq.append(elem)
        assert len(seq) == seq_len
        return seq

    def peek_state(self, seq_len, state):
        return self.peek_and_append(self.StateIndex, seq_len, state, state - state)






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

    def add(self, state, action, reward, done, next_state, *extra_pt_tensor):
        for i in range(self.n):
            extra = [pt[i] for pt in extra_pt_tensor]
            self.episode_collectors[i].add(state[i], action[i], reward[i], done[i], *extra)

    def has_full_batch(self, episodes_count):
        assert episodes_count > 0
        for ec in self.episode_collectors:
            if ec.episodes_count() <= 0:
                return False
        return sum([ec.episodes_count() for ec in self.episode_collectors]) >= episodes_count

    def roll_batch(self):
        results = []
        for i in range(self.n):
            # ep_extras: [(pt1,pt2,pt3...), (tt1,tt2,tt3,...), ...]
            while self.episode_collectors[i].episodes_count() > 0:
                results.append(self.episode_collectors[i].roll())
        return results

    def roll_batch_with_index(self):
        results = {}
        for i in range(self.n):
            j = 0
            # ep_extras: [(pt1,pt2,pt3...), (tt1,tt2,tt3,...), ...]
            while self.episode_collectors[i].episodes_count() > 0:
                results[(i, j)] = self.episode_collectors[i].roll()
                j += 1
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
            new_states.append(x)
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