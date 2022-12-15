#pip install --upgrade gym
#pip install gym[classic_control]

import gym
env = gym.make('CartPole-v1',render_mode="rgb_array")

obs = env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())