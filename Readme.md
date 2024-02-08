# policy gradient
有个非常重要的点是，计算discount reward时，一轮结束（得分1或-1）时将累计reward清0. 
karpathy代码注释里有说，是针对pong特地设置的：reset the sum, since this was a game boundary (pong specific!)

## karpathy.pong.py
karpathy实现。

## policy_gradient.py
### pg.episode.norm_discount_reward
Config(model_dir='models', model='pg.episode.norm_discount_reward', env_id='PongDeterministic-v0', game_visible=False, device='cpu', wandb=None, run_in_notebook=False, n_envs=8, lr=0.0001, hidden_dim=200, c_entropy=0.0, gamma=0.99, batch_size=100000, epoch_episodes=10, epoch_save=500, max_epoch=1000000, target_reward=20)

完全按照karpathy实现：state计算前后diff，环境action数量从6改为2（仅2、3有效），对图片进行prepro处理，使用MLP网络，每10个episode训练一次，计算discount reward，且进行normalize，loss函数为 -log_p * reward

学习率可以进一步提升为1e-3，训练可以快很多。HOME GPU训练

### pg.episode.no_discount_reward.no_norm
Config(model_dir='models', model='pg.episode.no_discount_reward.no_norm', env_id='PongDeterministic-v0', game_visible=False, device='cuda', wandb=None, run_in_notebook=True, n_envs=8, lr=0.0001, hidden_dim=200, c_entropy=0.0, gamma=1.0, max_epoch=1000000, epoch_episodes=10, epoch_save=500, target_reward=20)

see log https://www.kaggle.com/code/jofsky/pg-pong/log?scriptVersionId=162123347

与pg.episode.norm_discount_reward区别：不计算discount reward，不对reward normalize，其他配置完全一样。

训练慢很多。
