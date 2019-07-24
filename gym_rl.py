import gym
import time
import tensorflow as tf

from stable_baselines.common.policies import CnnPolicy, MlpLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.atari_wrappers import FrameStack
from stable_baselines import PPO2


class Agent:
    def __init__(self):
        self.env = None
        self.model = None

    def create_env(self, game, envs, render=False, sleep=0.):
        env = gym.make(game)
        # env = FrameStack(env, 4)
        env = CustomGym(env, render=render, sleep=sleep)
        self.env = SubprocVecEnv([lambda: env for i in range(envs)])

    def create_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config):
            self.model = PPO2(policy=MlpPolicy, env=self.env, n_steps=8192, nminibatches=8, lam=0.95, gamma=0.99,
                              noptepochs=4, ent_coef=0.001, learning_rate=lambda _: 2e-5, cliprange=lambda _: 0.2,
                              verbose=1, tensorboard_log="gym_logs")

    def train(self, timesteps, loops, name="agent"):
        for i in range(loops):
            self.model.learn(timesteps)
            self.model.save(name+str(i))

    def evaluate(self, timesteps, agent_name):
        self.model = PPO2.load(agent_name)
        obs = self.env.reset()
        for i in range(timesteps):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env.step(action)


class CustomGym(gym.Wrapper):
    def __init__(self, env, render=False, sleep=0.):
        super(CustomGym, self).__init__(env)
        self.render = render
        self.sleep = sleep

    def step(self, action):
        if self.render:
            self.env.render()
            if self.sleep >0:
                time.sleep(self.sleep)
        obs, rew, done, info = self.env.step(action)
        #print("Reward: "+str(rew))
        return obs, rew, done, info

if __name__ == "__main__":
    agent = Agent()
    agent.create_env('Breakout-v0', 8, render=False, sleep=0.)
    agent.create_model()
    agent.train(400000, 5)
    agent.create_env('Breakout-v0', 1, render=True, sleep=0.1)
    agent.evaluate(10000, "agent4.pkl")