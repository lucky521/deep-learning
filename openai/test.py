import gym, sys
env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)

sys.exit()
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print('episode finished after {} timesteps'.format(t+1))
            break


