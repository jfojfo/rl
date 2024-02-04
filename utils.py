from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


class Config(SimpleNamespace):
    pass


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

    def update_global_step(self, global_step):
        self.global_step = global_step

    def check_steps(self):
        return self.global_step % self.steps_to_log == 0


def wandb_login():
    import wandb
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    secret_wandb = user_secrets.get_secret("WANDB_API_KEY")

    wandb.login(key=secret_wandb)


if __name__ == '__main__':
    MySummaryWriter(0, 100)
