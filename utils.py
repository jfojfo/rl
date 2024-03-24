from types import SimpleNamespace
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv


normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape} {normal_repr(self)}"


class Config(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**{k: self._convert(v) for k, v in kwargs.items()})

    @staticmethod
    def _convert(value):
        if isinstance(value, dict):
            return Config(**value)
        elif isinstance(value, list):
            return [Config._convert(item) for item in value]
        else:
            return value


def normalize(data, axis=None):
    return (data - data.mean(axis, keepdims=True)) / (data.std(axis, keepdims=True) + 1e-8)


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2, ::2, 0:1] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float32).transpose(2, 0, 1)


def batch_prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[:, 35:195] # crop
  I = I[:, ::2, ::2, 0:1] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float32).transpose(0, 3, 1, 2)


def grey_crop_resize_batch(state, env_id_list=None):  # deal with batch observations
    states = []
    for i, ob in enumerate(state):
        array_3d = grey_crop_resize(ob, env_id_list[i] if env_id_list is not None else None)
        array_4d = np.expand_dims(array_3d, axis=0)
        states.append(array_4d)
    states_array = np.vstack(states) # turn the stack into array
    return states_array # B*C*H*W


crop_rect = {
    'Pong': (0, 160, 34, 194),
    'Breakout': (0, 160, 32, 196),
}
def grey_crop_resize(state, env_id=None): # deal with single observation
    img = Image.fromarray(state)
    grey_img = img.convert(mode='L')
    left, right, top, bottom = 0, img.width, 0, img.height
    if env_id is not None:
        for k, v in crop_rect.items():
            if env_id.startswith(k):
                left, right, top, bottom = v
                break
    cropped_img = grey_img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((80, 80))
    array_2d = np.asarray(resized_img)
    array_3d = np.expand_dims(array_2d, axis=0)
    return array_3d / 255. # C*H*W


# def grey_crop_resize(state): # deal with single observation
#     img = Image.fromarray(state)
#     grey_img = img.convert(mode='L')
#     resized_img = grey_img.resize((80, 80))
#     array_2d = np.asarray(resized_img)
#     array_3d = np.expand_dims(array_2d, axis=0)
#     return array_3d / 255. # C*H*W


class ActionModifierWrapper(gym.Wrapper):
    def __init__(self, n_actions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_actions = n_actions

    def step(self, action):
        # Modify the action before passing it to the environment
        if action >= self.n_actions:
            action = 0  # NOOP
        return self.env.step(action)


class KeepInfoClipRewardEnv(ClipRewardEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['reward'] = reward
        return obs, self.reward(reward), terminated, truncated, info


class KeepInfoEpisodicLifeEnv(EpisodicLifeEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        lives = info['lives']
        info['terminated'] = terminated and lives == 0
        return obs, reward, terminated, truncated, info


class NPFrameStack(gym.wrappers.FrameStack):
    def observation(self, observation):
        return np.array(super().observation(observation))


class NormObsWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, *extra = super().reset(**kwargs)
        return obs / 255.0, *extra

    def step(self, action):
        obs, *extra = super().step(action)
        return obs / 255.0, *extra


class ExpandDimWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        return obs[np.newaxis, ...]

    def step(self, action):
        obs, reward, done, _, info = super().step(action[0])
        return obs[np.newaxis, ...], np.array([reward]), np.array([done]), info


class EnvPoolWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs / 255.0

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        return obs / 255.0, reward, done, info


class CropFrameWrapper(gym.Wrapper):
    CropRect = {
        'Pong': (0, 160, 34, 194),
        'Breakout': (0, 160, 32, 196),
    }

    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        env_id = self.env_id
        for k, v in self.CropRect.items():
            if env_id.startswith(k):
                left, right, top, bottom = v
                break
        return obs


class MySummaryWriter(SummaryWriter):
    def __init__(self, step=0, steps_to_log=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_step = step
        self.steps_to_log = steps_to_log
        self.summary = {}
        self.summary_images = {}
        self.summary_cum = {}

    def update_global_step(self, global_step):
        self.global_step = global_step

    def check_steps(self):
        return self.global_step % self.steps_to_log == 0

    def summary_script_content(self, script_path):
        with open(script_path, 'r') as script_file:
            content = script_file.read()
            content = f'```python\n{content}\n```'
            self.summary_text(script_path, content)

    def summary_text(self, key, content):
        self.add_text(key, content, self.global_step)

    def summary_grad(self, optimizer, params, losses):
        if not self.check_steps():
            return
        summary = self.summary
        for name, loss in losses.items():
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grad_mean = params.grad.mean()
            grad_max = params.grad.max()
            self.stash_summary('Grad', {name: grad_mean, f'{name}:max': grad_max})

    def summary_loss(self, losses, weight=None):
        self.stash_summary('Loss', losses, weight)
        # for name, loss in losses.items():
        #     k = f'Loss/{name}'
        #     if weight is not None:
        #         if k not in self.summary_cum:
        #             self.summary_cum[k] = [0, 0]
        #         self.summary_cum[k][0] += loss * weight
        #         self.summary_cum[k][1] += weight
        #     else:
        #         self.summary[k] = loss

    def summary_attns(self, attns):
        if not self.check_steps():
            return
        for i, attn in enumerate(attns):
            self.summary_images[f'Attention/{i}'] = attn

    def stash_summary(self, tag, sum_dict, weight=None):
        summary = self.summary.get(tag, {})
        self.summary[tag] = summary
        for k, v in sum_dict.items():
            if weight is None:
                summary[k] = v
            else:
                if k not in summary:
                    summary[k] = [0, 0]
                summary[k][0] += v * weight
                summary[k][1] += weight

    def write_summary(self):
        for tag, summary in self.summary.items():
            for name, v in summary.items():
                if type(v) is list:
                    v = v[0] / v[1]
                self.add_scalar(f'{tag}/{name}', v, self.global_step)

        for k, img in self.summary_images.items():
            self.add_image(k, img, self.global_step)

        self.summary = {}
        self.summary_images = {}

    def pop_summary(self, tag):
        ret = {}
        summary = self.summary.pop(tag, {})
        for name, v in summary.items():
            if type(v) is list:
                v = v[0] / v[1]
            ret[name] = v
        return ret

    def flush_summary(self):
        self.write_summary()


# along axis 0: seq axis
def lookback_mask(dones):
    mask = 1 - dones
    mask[-1] = 1  # ... 0 -> ... 1
    mask = mask[::-1]
    mask = np.minimum.accumulate(mask, axis=0)
    mask = mask[::-1]
    return mask.copy()  # return copy to prevent reverse index error when converting torch tensor


def wandb_login():
    import wandb
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    secret_wandb = user_secrets.get_secret("WANDB_API_KEY")

    wandb.login(key=secret_wandb)

def wandb_init(name, cfg):
    if name is not None:
        import wandb
        wandb_login()
        wandb.init(project=name, sync_tensorboard=True, config=cfg)

def wandb_finish(name):
    if name is not None:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    MySummaryWriter(0, 100)
