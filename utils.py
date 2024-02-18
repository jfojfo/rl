from types import SimpleNamespace
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


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


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2, ::2, 0:1] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float32)


def batch_prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[:, 35:195] # crop
  I = I[:, ::2, ::2, 0:1] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float32)


def grey_crop_resize_batch(state):  # deal with batch observations
    states = []
    for i in state:
        array_3d = grey_crop_resize(i)
        array_4d = np.expand_dims(array_3d, axis=0)
        states.append(array_4d)
    states_array = np.vstack(states) # turn the stack into array
    return states_array # B*C*H*W


def grey_crop_resize(state): # deal with single observation
    img = Image.fromarray(state)
    grey_img = img.convert(mode='L')
    left = 0
    top = 34  # empirically chosen
    right = 160
    bottom = 194  # empirically chosen
    cropped_img = grey_img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((84, 84))
    array_2d = np.asarray(resized_img)
    array_3d = np.expand_dims(array_2d, axis=0)
    return array_3d / 255. # C*H*W


class MySummaryWriter(SummaryWriter):
    def __init__(self, step=0, steps_to_log=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_step = step
        self.steps_to_log = steps_to_log
        self.summary = {}

    def update_global_step(self, global_step):
        self.global_step = global_step

    def check_steps(self):
        return self.global_step % self.steps_to_log == 0

    def summary_script_content(self, script_path):
        with open(script_path, 'r') as script_file:
            content = script_file.read()
            content = f'```python\n{content}\n```'
            self.add_text(script_path, content, self.global_step)

    def summary_grad(self, optimizer, params, losses):
        if not self.check_steps():
            return
        summary = self.summary
        for name, loss in losses.items():
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grad_mean = params.grad.mean()
            summary[f'Grad/{name}'] = grad_mean.item()

    def summary_loss(self, losses):
        summary = self.summary
        for name, loss in losses.items():
            summary[f'Loss/{name}'] = loss.item()

    def write_summary(self):
        for k, v in self.summary.items():
            self.add_scalar(k, v, self.global_step)
        self.summary = {}


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
