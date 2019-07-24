import tensorflow as tf
from stable_baselines import PPO2 as stable_PPO2
from common.sonic_env import *
from functools import partial

from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.policies import CnnPolicy

class PPO2:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.env_fns = []
        self.env_names = []
        self.environs = ['SpringYardZone.Act3', 'SpringYardZone.Act2', 'GreenHillZone.Act3', 'GreenHillZone.Act1',
                         'StarLightZone.Act2', 'StarLightZone.Act1', 'MarbleZone.Act2', 'MarbleZone.Act1',
                         'MarbleZone.Act3', 'ScrapBrainZone.Act2', 'LabyrinthZone.Act2', 'LabyrinthZone.Act1',
                         'LabyrinthZone.Act3', 'SpringYardZone.Act1', 'GreenHillZone.Act2', 'StarLightZone.Act3',
                         'ScrapBrainZone.Act1']
        self.environsv2 = ['1Player.Axel.Level1']

    def create_envs(self, game_name, state_name, num_env, render):
        for state in state_name:
            for i in range(num_env):
                print()
                self.env_fns.append(partial(make_env, game=game_name, state=state, render=render))
                self.env_names.append(game_name + '-' + state)
        self.env = SubprocVecEnv(self.env_fns)


    def train(self, game, state, num_e=1, n_timesteps=200000, save='ppo-model'):
        self.create_envs(game_name=game, state_name=state, num_env=num_e, render=self.FLAGS.render)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config):
            self.model = stable_PPO2(policy=CnnPolicy, env=SubprocVecEnv(self.env_fns), n_steps=8192, nminibatches=8,
                                     lam=0.95, gamma=0.99, noptepochs=4, ent_coef=0.001, learning_rate=lambda _: 2e-5,
                                     cliprange=lambda _: 0.2, verbose=1, tensorboard_log=self.FLAGS.logdir)

        self.model.learn(n_timesteps)
        self.model.save(save+'1')
        self.model.learn(n_timesteps)
        self.model.save(save+'2')
        self.model.learn(n_timesteps)
        self.model.save(save+'3')
        self.model.learn(n_timesteps)
        self.model.save(save+'4')
        self.model.learn(n_timesteps)
        self.model.save(save+'5')


    def retrain(self, game, state, num_e=1, n_timesteps=2000, save='my-model'):
        self.create_envs(game_name=game, state_name=state, num_env=num_e, render=self.FLAGS.render)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config):
            self.model = stable_PPO2.load(self.FLAGS.model, policy=CnnPolicy, env=SubprocVecEnv(self.env_fns),
                                          n_steps=8192, nminibatches=8, lam=0.95, gamma=0.99, noptepochs=4,
                                          ent_coef=0.001, learning_rate=lambda _: 2e-5, cliprange=lambda _: 0.2,
                                          verbose=1, tensorboard_log=self.FLAGS.logdir)

        self.model.learn(n_timesteps)
        self.model.save(save + '1')
        self.model.learn(n_timesteps)
        self.model.save(save + '2')
        self.model.learn(n_timesteps)
        self.model.save(save + '3')
        self.model.learn(n_timesteps)
        self.model.save(save + '4')
        self.model.learn(n_timesteps)
        self.model.save(save + '5')

    def evaluate(self, game, state, num_e=1, num_steps=14400):
        self.create_envs(game_name=game, state_name=state, num_env=num_e, render=self.FLAGS.render)
        self.model = stable_PPO2.load(self.FLAGS.model, SubprocVecEnv(self.env_fns), policy=CnnPolicy,
                                      tensorboard_log=self.FLAGS.logdir)
        episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
        obs = self.env.reset()
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            actions, _states = self.model.predict(obs)
            # # here, action, rewards and dones are arrays
            # # because we are using vectorized env
            obs, rewards, dones, info = self.env.step(actions)

            # Stats
            for i in range(self.env.num_envs):
                episode_rewards[i][-1] += rewards[i]
                if dones[i]:
                    episode_rewards[i].append(0.0)

        mean_rewards =  [0.0 for _ in range(self.env.num_envs)]
        n_episodes = 0
        for i in range(self.env.num_envs):
            mean_rewards[i] = np.mean(episode_rewards[i])
            n_episodes += len(episode_rewards[i])

        # Compute mean reward
        mean_reward = np.mean(mean_rewards)
        print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

        return mean_reward