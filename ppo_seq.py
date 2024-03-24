import os
import random
import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import optim
import torch.nn.functional as F
from model import *
from utils import *
from collector import *
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


config = {
    'model_net': 'transformer',  # mlp, cnn, cnn3d, transformer, transformercnn, dt
    'model': 'ppo.seq.transformer.good',
    'model_dir': 'models',
    'env_id_list': ['Breakout-v5'] * 8,
    'envpool': True,
    # 'env_id_list': ['BreakoutNoFrameskip-v4'] * 2,
    # 'envpool': False,
    # 'env_id_list': ['PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'MsPacmanNoFrameskip-v4'],
    'game_visible': False,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'wandb': 'ppo.seq.cnn.good',
    # 'run_in_notebook': True,
    'wandb': None,
    'run_in_notebook': False,

    'lr': 1.5e-4,
    'hidden_dim': 512,  # hidden size, linear units of the output layer
    'c_critic': 1.0,  # critic coefficient
    'c_entropy': 0.01,  # entropy coefficient
    'gamma': 0.99,
    'lam': 0.95,
    'surr_clip': 0.2,
    'value_clip': 0.2,
    'grad_norm': 10.0,

    'n_actions': 6,
    'optimise_times': 4,  # optimise times per epoch
    'max_batch_size': 1000,  # mlp 10w, cnn 3w, cnn3d 3k, transformer 1k
    'chunk_percent': 1/8,  # split into chunks, do optimise per chunk
    'chunk_size': None,
    'seq_len': 65,
    'epoch_size': 256,
    'epoch_episodes': 8,
    'epoch_save': 900,
    'max_epoch': 30000,
    'max_episode_steps': 8000,
    'target_reward': 20000000,
    'diff_state': False,
    'shuffle': True,

    'transformer': {
        'num_blocks': 1,
        'dim_input': 256,  # dim_head * num_heads
        'dim_head': 32,
        'num_heads': 8,
        'state_dim': 84 * 84,
        'action_dim': 6,
        'reward_dim': 1,
        # 'window_size': 256,  # seq_len - 1
        'layer_norm': 'pre',  # pre/post
        'positional_encoding': 'none',  # relative/learned/rotary
        'dropout': 0.0,
    },
}
cfg = Config(**config)
cfg.transformer.device = cfg.device
cfg.transformer.window_size = cfg.seq_len - 1
# cfg.n_actions = cfg.transformer.action_dim = 4 # env_.action_space.n



def set_action_dim(env):
    cfg.n_actions = cfg.transformer.action_dim = env.action_space.n


def discount_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    returns = [0] * len(rewards)
    next_value = next_value
    gae = 0
    for i in reversed(range(len(rewards))):
        # if rewards[i] != 0:
        #     gae = 0
        #     next_value = 0
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + lam * gamma * gae * (1 - dones[i])
        returns[i] = values[i] + gae
        next_value = values[i]
    return returns


def loss_ppo(trainer, data_loader, states, actions, dones, old_log_probs, old_values, rewards, advantages, *extra):
    params = trainer.get_model_params(data_loader, states, *extra, is_train=True)
    _, log_probs, values, entropy = trainer.predict(*params, actions, is_train=True)

    # todo: normalize?
    # advantages = rewards - values.detach()
    # no norm for transformer, will cause NaN
    advantages = normalize(advantages)
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - cfg.surr_clip, 1.0 + cfg.surr_clip) * advantages
    actor_loss = -torch.min(surr1, surr2).sum() / len(data_loader)

    actor_clip_fracs = ((ratio - 1.0).abs() > cfg.surr_clip).float().mean()
    trainer.writer.stash_summary('Monitor', {'actor_clip_fracs': actor_clip_fracs}, weight=len(data_loader))

    critic_loss = (values - rewards).pow(2)
    if cfg.value_clip is not None:
        # clipped_values = old_values + torch.clamp(values - old_values, -cfg.value_clip, cfg.value_clip)
        # clipped_critic_loss = (clipped_values - rewards).pow(2)
        # critic_loss = torch.max(critic_loss, clipped_critic_loss)

        critic_clip_fracs = ((values - old_values).abs() > cfg.value_clip).float().mean()
        trainer.writer.stash_summary('Monitor', {'critic_clip_fracs': critic_clip_fracs}, weight=len(data_loader))

    critic_loss = 0.5 * critic_loss.sum() / len(data_loader)
    entropy_loss = -entropy.sum() / len(data_loader)
    loss = actor_loss + critic_loss * cfg.c_critic + entropy_loss * cfg.c_entropy
    return {'critic_loss': critic_loss, 'actor_loss': actor_loss, 'entropy_loss': entropy_loss, 'loss': loss}


class Train:
    def __init__(self):
        self.writer = MySummaryWriter(0, max(1, cfg.epoch_save // 4), comment=f'.{cfg.model}')

    @staticmethod
    def diff_state(state, last_state):
        return state if not cfg.diff_state else (state - last_state)

    @staticmethod
    def get_stack_frame_num():
        if cfg.model_net.startswith('transformer') or cfg.model_net.startswith('cnn3d'):
            return 1
        else:
            return 4

    @staticmethod
    def make_env_fn(env_id, seed=None, is_train=True, **kwargs):
        if cfg.envpool:
            def _init():
                import envpool
                env = envpool.make(
                    cfg.env_id_list[0],
                    env_type="gymnasium",
                    num_envs=1,
                    max_episode_steps=cfg.max_episode_steps,
                    episodic_life=is_train,
                    reward_clip=is_train,
                    stack_num=Train.get_stack_frame_num(),
                    seed=seed if seed is not None else np.random.randint(0, 10000),
                )
                env = EnvPoolWrapper(env)
                return env
            return _init
        else:
            def _init():
                env = gym.make(env_id, max_episode_steps=cfg.max_episode_steps, **kwargs)
                n_actions = env.action_space.n
                if env_id.startswith('MsPacman'):
                    n_actions = 5
                env = ActionModifierWrapper(n_actions, env)
                if is_train:
                    env = NoopResetEnv(env, noop_max=30)
                env = MaxAndSkipEnv(env, skip=4)
                if is_train:
                    env = KeepInfoEpisodicLifeEnv(env)
                    env = KeepInfoClipRewardEnv(env)
                env = gym.wrappers.ResizeObservation(env, (84, 84))
                env = gym.wrappers.GrayScaleObservation(env)
                env = NormObsWrapper(env)
                env = NPFrameStack(env, Train.get_stack_frame_num())
                env.seed(seed)  # Apply a unique seed to each environment
                return env
            return _init

    @staticmethod
    def make_envs(is_train=True):
        if cfg.envpool:
            import envpool
            envs = envpool.make(
                cfg.env_id_list[0],
                env_type="gymnasium",
                num_envs=len(cfg.env_id_list),
                max_episode_steps=cfg.max_episode_steps,
                episodic_life=is_train,
                reward_clip=is_train,
                stack_num=Train.get_stack_frame_num(),
                seed=np.random.randint(0, 10000),
            )
            envs = EnvPoolWrapper(envs)
            return envs
        else:
            envs = [Train.make_env_fn(env_id, seed=np.random.randint(0, 10000), render_mode='human' if cfg.game_visible else 'rgb_array') for env_id in cfg.env_id_list]
            envs = SubprocVecEnv(envs)
            return envs

    @staticmethod
    def make_test_env(**kwargs):
        env_id = cfg.env_id_list[0]
        fn = Train.make_env_fn(env_id, is_train=False, **kwargs)
        env_test = fn()
        if not cfg.envpool:
            env_test = ExpandDimWrapper(env_test)
        return env_test

    @staticmethod
    def get_model(cfg, num_outputs=0):
        if cfg.model_net == 'mlp':
            return MLPModel(cfg, num_outputs)
        elif cfg.model_net == 'cnn':
            return CNNModel2(cfg, num_outputs)
        elif cfg.model_net == 'cnn3d':
            return CNN3dModel2(cfg, num_outputs)
        elif cfg.model_net == 'cnn3d2d':
            return CNN3d2dModel(cfg, num_outputs)
        elif cfg.model_net == 'transformer':
            return TrModel(cfg.transformer, num_outputs)
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
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            print(f'Model loaded, starting from epoch {epoch}')
        return epoch

    @staticmethod
    def save_model(model, optimizer, epoch, avg_reward):
        name = "%s_%+.3f_%d.pth" % (cfg.model, avg_reward, epoch)
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
        lr = max(cfg.lr * (1.0 - (epoch / cfg.max_epoch)), cfg.lr / 10)
        self.writer.add_scalar("lr", lr, epoch)
        self.optimizer.param_groups[0]["lr"] = lr

    def end_epoch_optimize(self, epoch, envs):
        self.collector.clear_all()
        return envs.reset()

    def get_data_collector(self):
        return BatchEpisodeCollector(len(cfg.env_id_list))

    def train_eval(self):
        return self.avg_reward

    def get_model_params(self, collector, state, *_, is_train=False):
        if is_train:
            return state,
        else:
            # return tuple
            return torch.FloatTensor(state).to(cfg.device),

    def predict(self, state, action=None, is_train=False):
        dist, value = self.model(state)
        value = value.squeeze(1)
        if is_train:
            log_prob = dist.log_prob(action)
            return action, log_prob, value, dist.entropy()
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.detach().cpu().numpy().astype(np.int32), log_prob.detach().cpu().numpy(), value.detach().cpu().numpy()

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
            print(f'Run {(epoch - 1) * cfg.epoch_size + i + 1}, steps {ep_steps}, reward {ep_reward_sum}, cum_reward {cum_reward:.3f}')
            total_samples += len(ep_rewards)

            self.recalc_episode(episode)

        avg_reward /= len(episodes)
        self.writer.add_scalar('Reward/epoch_reward_avg', avg_reward, self.writer.global_step)
        self.writer.add_scalar('Reward/env_1_reward', sum(episodes[0][2]), self.writer.global_step)
        self.cum_reward = cum_reward
        self.avg_reward = avg_reward
        return total_samples

    # def process_episodes(self, episodes):
    #     if not hasattr(self, 'cum_reward'):
    #         self.cum_reward = {i: None for i in range(len(cfg.env_id_list))}
    #     total_samples = 0
    #     avg_reward = {}
    #     avg_reward = {i: [] for i in range(len(cfg.env_id_list))}
    #     cum_reward = self.cum_reward
    #     epoch = self.writer.global_step
    #     for (i, j), episode in episodes.items():
    #         ep_states, ep_actions, ep_rewards, ep_dones, *_ = episode
    #         ep_reward_sum, ep_steps = sum(ep_rewards), len(ep_rewards)
    #         avg_reward[i].append(ep_reward_sum)
    #         cum_reward[i] = ep_reward_sum if cum_reward[i] is None else cum_reward[i] * 0.99 + ep_reward_sum * 0.01
    #         print(f'Epoch {epoch}, {cfg.env_id_list[i]} steps {ep_steps}, reward {ep_reward_sum}, cum_reward {cum_reward[i]:.3f}')
    #         total_samples += len(ep_rewards)

    #         self.recalc_episode(episode)

    #     for i, avg_reward_list in avg_reward.items():
    #         avg = sum(avg_reward_list) / len(avg_reward_list)
    #         self.writer.add_scalar(f'Reward/epoch_reward_avg/{cfg.env_id_list[i]}', avg, self.writer.global_step)
    #     self.writer.add_scalar('Reward/env_1_reward', avg_reward[0][0], self.writer.global_step)
    #     self.cum_reward = cum_reward
    #     self.avg_reward = sum(avg_reward[0]) / len(avg_reward[0])
    #     return total_samples

    def recalc_episode(self, episode):
        ep_states, ep_actions, ep_rewards, ep_dones, ep_log_probs, ep_values, *_ = episode
        del episode[6:]
        ep_rewards = np.array(ep_rewards)
        ep_discount_rewards = discount_rewards_episodely(ep_rewards, cfg.gamma)
        ep_discount_rewards /= (np.abs(ep_discount_rewards).max() + 1e-8)
        # ep_discount_rewards = discount_gae_roundly(ep_rewards, ep_values, cfg.gamma, cfg.lam)
        # ep_rewards = normalize(ep_rewards)
        # ep_rewards = list(ep_rewards)
        # episode[2] = ep_rewards
        episode.insert(len(episode), list(ep_discount_rewards))
        ep_values = np.array([v.item() for v in ep_values])
        ep_advantages = ep_discount_rewards - ep_values
        episode.insert(len(episode), list(ep_advantages))

    def loss(self, data_loader, states, actions, dones, old_log_probs, old_values, rewards, advantages, *extra):
        return loss_ppo(self, data_loader, states, actions, dones, old_log_probs, old_values, rewards, advantages, *extra)

    def optimise_by_minibatch(self, model, optimizer, data_loader):
        acc_loss = {}
        optimizer.zero_grad()
        for states, actions, rewards, dones, *extra in data_loader.next_batch():
            loss_dict = self.loss(data_loader, states, actions, dones, *extra)

            # summary in the first iteration and zero_grad after that
            if data_loader.get_iter() == 0:
                params_feature = model.get_parameter('feature.linear.weight')
                self.writer.summary_grad(optimizer, params_feature, loss_dict)
                optimizer.zero_grad()

            loss_dict['loss'].backward()

            keys = set(acc_loss).union(loss_dict)
            acc_loss = {key: acc_loss.get(key, 0) + loss_dict.get(key, 0) for key in keys}
        total_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm)
        optimizer.step()
        self.writer.stash_summary('Grad', {'grad_norm': total_grad_norm}, weight=1)
        self.writer.summary_loss(acc_loss)

    def train(self, load_from):
        writer = self.writer
        if not cfg.run_in_notebook:
            writer.summary_script_content(__file__)
        else:
            writer.summary_text('', f'```python\n{In[-1]}\n```')

        envs = self.make_envs()
        set_action_dim(envs)
        num_outputs = cfg.n_actions # = cfg.transformer.action_dim = envs.action_space.n

        model = self.model = self.get_model(cfg, num_outputs).to(cfg.device)
        optimizer = self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)  # implements Adam algorithm
        epoch = self.load_model(load_from, model, optimizer)
        print(cfg)
        print(model)
        print(optimizer)

        collector = self.collector = self.get_data_collector()
        state = envs.reset()
        # state = grey_crop_resize_batch(state, cfg.env_id_list)
        last_state = state.copy()

        early_stop = False

        while not early_stop and epoch < cfg.max_epoch:
            state_ = self.diff_state(state, last_state)

            with torch.no_grad():
                params = self.get_model_params(collector, state_)
                action, *extra = self.predict(*params)

            # action = action.cpu().numpy()
            next_state, reward, done, info = envs.step(action)
            # next_state = grey_crop_resize_batch(next_state, cfg.env_id_list)

            # reset state to next_state when done
            for i, d in enumerate(done):
                if d:
                    state[i] = next_state[i]
            next_state_ = self.diff_state(next_state, state)
            if cfg.envpool:
                extra.append(info)
            else:
                extra.append({'terminated': np.array([d['terminated'] for d in info], dtype=np.int32),
                              'reward': np.array([d['reward'] for d in info])})
            collector.add(state_, action, reward, done, next_state_, *extra)

            last_state, state = state, next_state

            if collector.has_full_batch(cfg.epoch_size):
                epoch += 1
                self.begin_epoch_optimize(epoch)

                # episodes = collector.roll_batch_with_index()
                episodes = collector.roll_batch()
                total_samples = self.process_episodes(episodes)

                for _ in range(cfg.optimise_times):
                    # optimise every chunk_percent samples to accelerate training
                    chunk_size = cfg.chunk_size if cfg.chunk_size is not None else int(np.ceil(total_samples * cfg.chunk_percent))
                    chunk_loader = self.get_chunk_loader(episodes, chunk_size, lambda *d: d, cfg.shuffle)

                    for chunk in chunk_loader.next_batch():
                        # minibatch don't exceeds cfg.max_batch_size
                        minibatch_loader = self.get_minibatch_loader(chunk_loader, cfg.max_batch_size, self.data_fn, False, *chunk)
                        self.optimise_by_minibatch(model, optimizer, minibatch_loader)
                        monitor_summary = self.writer.pop_summary('Monitor')
                # write summary once after optimise_times to prevent duplicate
                self.writer.stash_summary('Monitor', monitor_summary)
                writer.write_summary()

                # need reset for training episodely
                reset_state = self.end_epoch_optimize(epoch, envs)
                if reset_state is not None:
                    # state = grey_crop_resize_batch(reset_state, cfg.env_id_list)
                    last_state = state.copy()

                if epoch % cfg.epoch_save == 0:
                    avg_reward = self.train_eval()
                    print(f'Epoch: {epoch} -> Reward: {avg_reward}, saving model')
                    self.writer.add_scalar('Reward/epoch_reward_avg', avg_reward, self.writer.global_step)
                    self.save_model(model, optimizer, epoch, avg_reward)

                    if avg_reward > cfg.target_reward:
                        early_stop = True

    def test_env(self, env, model):
        state = env.reset()
        # state = grey_crop_resize(state, env.spec.id)
        last_state = state.copy()

        collector = BatchEpisodeCollector(1)
        done = False
        total_reward = 0
        steps = 0
        while not done:
            state_ = self.diff_state(state, last_state)
            # state_ = state_[np.newaxis, ...]

            params = self.get_model_params(collector, state_)
            action, *extra = self.predict(*params)

            next_state, reward, done, _ = env.step(action)
            # next_state = grey_crop_resize(next_state, env.spec.id)
            # next_state_ = next_state[np.newaxis, ...]

            collector.add(state, action, reward, done, next_state)
            last_state = state
            state = next_state
            total_reward += reward[0]
            steps += 1
        collector.clear_all()
        self.end_epoch_optimize(0, env)
        return total_reward, steps

    def eval(self, load_from=None):
        assert load_from is not None

        env_test = self.make_test_env(render_mode='human')
        # num_outputs = env_test.action_space.n
        set_action_dim(env_test)
        num_outputs = cfg.n_actions # = cfg.transformer.action_dim = env_test.action_space.n
        model = self.get_model(cfg, num_outputs).to(cfg.device)
        model.eval()

        self.load_model(load_from, model, None)

        while True:
            reward, steps = self.test_env(env_test, model)
            print(f"steps {steps}, reward {reward}")

    def analysis(self, load_from=None):
        env_test = self.make_test_env()
        num_outputs = cfg.n_actions # = cfg.transformer.action_dim = env_test.action_space.n
        model = self.get_model(cfg, num_outputs).to(cfg.device)
        self.load_model(load_from, model, None)
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Model/{name}', param.flatten(), self.writer.global_step, yscale="percentage")


class SeqTrain(Train):
    def process_episodes(self, seq_data):
        with torch.no_grad():
            params = self.get_model_params(self.collector, self.collector.next_state)
            next_action, next_log_prob, next_value, *_ = self.predict(*params)

        self.update_1_env(seq_data)
        del seq_data[6:]

        sq_states, sq_actions, sq_rewards, sq_dones, sq_log_probs, sq_values = seq_data
        sq_values_scaled = reverse_scaled_sigmoid(torch.as_tensor(np.array(sq_values))).numpy()
        next_value_scaled = reverse_scaled_sigmoid(torch.as_tensor(next_value)).numpy()
        sq_discount_rewards = discount_gae(sq_rewards, sq_dones, sq_values_scaled, next_value_scaled, cfg.gamma, cfg.lam)
        sq_discount_rewards = list(scaled_sigmoid(torch.as_tensor(np.array(sq_discount_rewards))).numpy())

        seq_data.insert(len(seq_data), sq_discount_rewards)
        sq_advantages = [discount_reward - value for discount_reward, value in zip(sq_discount_rewards, sq_values)]
        # sq_advantages = list(normalize(np.array(sq_advantages)))
        seq_data.insert(len(seq_data), sq_advantages)

        total_samples = len(sq_states) * len(sq_states[0])
        return total_samples

    def update_1_env(self, seq_data):
        offset = len(seq_data[0]) - cfg.epoch_size
        seq_data = [d[offset:] for d in seq_data]

        if not hasattr(self, 'total_runs'):
            self.total_runs = np.zeros(len(cfg.env_id_list), dtype=np.int32)
            self.total_steps = np.zeros(len(cfg.env_id_list), dtype=np.int32)
            self.total_rewards = np.zeros(len(cfg.env_id_list), dtype=np.float32)

        infos = seq_data[-1]
        terminated = [info['terminated'] for info in infos]
        real_rewards = [info['reward'] for info in infos]

        for reward, term in zip(real_rewards, terminated):
            self.total_steps += 1
            self.total_rewards += reward
            self.total_runs += term
            for i in np.where(term == 1)[0]:
                env_name = cfg.env_id_list[i]
                print(f'Epoch {self.writer.global_step} Env {i+1} Run {self.total_runs[i]}: steps {self.total_steps[i]}, Reward {self.total_rewards[i]}, {env_name}')
                self.writer.add_scalar(f'Reward/{i}/{env_name}', self.total_rewards[i], self.writer.global_step)
                self.writer.add_scalar(f'Steps/{i}/{env_name}', self.total_steps[i], self.writer.global_step)
            self.total_steps *= (1 - term)
            self.total_rewards *= (1 - term)

    def get_data_collector(self):
        return MultiStepCollector(cfg.epoch_size, len(cfg.env_id_list))

    def get_chunk_loader(self, seq_data, chunk_size, data_fn, random=False):
        return SqDataGenerator(seq_data, chunk_size, data_fn, random)

    def get_minibatch_loader(self, chunk_loader, max_batch_size, data_fn, random, *chunk):
        return BaseGenerator(max_batch_size, data_fn, random, *chunk)

    def end_epoch_optimize(self, epoch, envs):
        return None

    def train_eval(self):
        if not hasattr(self, 'env_test'):
            self.env_test = self.make_test_env()
        avg_reward = 0.0
        episodes = 10
        for _ in range(episodes):
            reward, steps = self.test_env(self.env_test, self.model)
            avg_reward += reward
        return avg_reward / episodes


class SeqTrainCNN3d(SeqTrain):
    def get_data_collector(self):
        return MultiStepCollector(cfg.epoch_size, len(cfg.env_id_list), lookback=cfg.seq_len - 1)

    def get_model_params(self, collector, state, *_, is_train=False):
        if is_train:
            return state,
        else:
            state_ = collector.peek_state(cfg.seq_len, state)
            # return tuple
            return torch.FloatTensor(state_).to(cfg.device),

    def get_chunk_loader(self, seq_data, chunk_size, data_fn, random=False):
        lookback = cfg.seq_len - 1
        offset = len(seq_data[0]) - cfg.epoch_size
        return StateSqDataGenerator(seq_data, chunk_size, lookback, offset, data_fn, random)


class SeqTrainTransformer(SeqTrain):
    def process_episodes(self, seq_data):
        infos = seq_data[-1]
        terminated = [info['terminated'] for info in infos]
        ret = super().process_episodes(seq_data)
        # replace with real terminated after super's process_episodes (need to calc advantage)
        seq_data[Collector.DoneIndex] = terminated
        return ret

    def get_data_collector(self):
        return TrMultiStepCollector(cfg.epoch_size, len(cfg.env_id_list), lookback=cfg.transformer.window_size)

    def get_chunk_loader(self, seq_data, chunk_size, data_fn, random=False):
        offset = len(seq_data[0]) - cfg.epoch_size
        return LookBackBatchChunkDataGenerator(chunk_size, cfg.transformer.window_size, offset, data_fn, *seq_data)

    def get_minibatch_loader(self, chunk_loader, max_batch_size, data_fn, random, *chunk):
        return LookBackBatchSeqDataGenerator(max_batch_size, cfg.transformer.window_size, chunk_loader.get_dynamic_offset(), data_fn, *chunk)

    def data_fn(self, *data):
        ret = super().data_fn(*data[:-1])
        mask = torch.as_tensor(np.array(data[-1]), dtype=torch.bool).to(cfg.device)  # requires_grad=False
        return ret + (mask,)

    def get_model_params(self, collector, state, mask=None, is_train=False):
        if is_train:
            data_loader = collector
            # lookback = data_loader.get_dynamic_offset()
            lookback = cfg.transformer.window_size
            query_mask = mask
            key_mask = mask
            return state, query_mask, key_mask, lookback, data_loader.get_iter(), data_loader
        else:
            state_ = state
            mask = collector.peek_mask(cfg.seq_len)
            pt_state = torch.as_tensor(state_, dtype=torch.float32).unsqueeze(1).to(cfg.device)
            pt_query_mask = None
            pt_key_mask = torch.as_tensor(mask, dtype=torch.bool).to(cfg.device)
            return pt_state, pt_query_mask, pt_key_mask

    def predict(self, state, query_mask, key_mask, lookback=0, n_iter=0, data_loader=None, action=None, is_train=False):
        if is_train:
            logits, value, attn_weight = self.model(state, query_mask, key_mask, is_train=is_train, lookback=lookback)
            if n_iter == 0:
                self.writer.summary_attns([attn[0].unsqueeze(0) for attn in attn_weight])

            state = state[:, lookback:]
            state, value, logits = data_loader.trans_back(state, value, logits)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            logits, value, attn_weight = self.model(state, query_mask, key_mask, is_train=is_train, lookback=lookback)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action, log_prob, value = action.squeeze(-1), log_prob.squeeze(-1), value.squeeze(-1)

        value = value.squeeze(-1)
        if is_train:
            return action, log_prob, value, entropy
        else:
            return action.detach().cpu().numpy().astype(np.int32), log_prob.detach().cpu().numpy(), value.detach().cpu().numpy()

    def begin_epoch_optimize(self, epoch):
        super().begin_epoch_optimize(epoch)
        self.model.clear_memory()

    def end_epoch_optimize(self, epoch, envs):
        ret = super().end_epoch_optimize(epoch, envs)
        self.model.clear_memory()
        return ret


def get_trainer():
    if cfg.model_net in ('cnn3d', 'cnn3d2d'):
        return SeqTrainCNN3d()
    elif cfg.model_net in ('transformer', 'transformercnn'):
        return SeqTrainTransformer()
    # elif cfg.model_net == 'dt':
    #     return TrainDT()
    else:
        return SeqTrain()


def train(load_from=None):
    wandb_init(cfg.wandb, cfg)
    get_trainer().train(load_from)
    wandb_finish(cfg.wandb)


def eval(load_from=None):
    get_trainer().eval(load_from)


def analysis(load_from=None):
    get_trainer().analysis(args.model)
    

if __name__ == "__main__":
    # python ppo_pong.py --eval --model models/ppo_pong.diff.5200.pth
    ap = argparse.ArgumentParser(description='Process args.')
    ap.add_argument('--eval', action='store_true', help='evaluate')
    ap.add_argument('--analysis', action='store_true', help='analysis')
    ap.add_argument('--model', type=str, default=None, help='model to load')
    args = ap.parse_args(args=[]) if cfg.run_in_notebook else ap.parse_args()

    os.makedirs(os.path.join(cfg.model_dir, cfg.model), mode=0o755, exist_ok=True)
    if args.eval:
        eval(args.model)
    elif args.analysis:
        analysis(args.model)
    else:
        train(args.model)
