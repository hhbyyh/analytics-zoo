import threading
import numpy as np
import tensorflow as tf
import time
import gym
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import pylab


# global variables for threading
episode = 0
scores = []

EPISODES = 2000

# This is A3C(Asynchronous Advantage Actor Critic) agent(global) for the Cartpole
# In this example, we use A3C algorithm
class A3CAgent:
    def __init__(self, state_size, action_size, env_name):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # get gym environment name
        self.env_name = env_name

        # these are hyper parameters for the A3C
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.discount_factor = .99
        self.hidden1, self.hidden2 = 24, 24
        self.threads = 1

        # create model for actor and critic network
        self.actor = self.build_actor()
        self.critic = self.build_critic()


    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='glorot_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic


    # make agents(local) and start training
    def train(self):
        # self.load_model('./save_model/cartpole_a3c.h5')
        agent = Agent(1, self.actor, self.critic, self.env_name, self.discount_factor,
                        self.action_size, self.state_size)

        agent.run()


# This is Agent(local) class for threading
class Agent():
    def __init__(self, index, actor, critic, env_name, discount_factor, action_size, state_size):

        self.states = []
        self.rewards = []
        self.actions = []

        self.index = index
        self.actor = actor
        self.critic = critic
        self.env_name = env_name
        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size

    # Thread interactive with environment
    def run(self):
        global episode
        env = gym.make(self.env_name)
        episodes = []
        while episode < EPISODES:
            state = env.reset()
            score = 0
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                score += reward

                self.memory(state, action, reward)

                state = next_state

                if done:
                    episode += 1
                    episodes.append(episode)
                    print("episode: ", episode, "/ score : ", score)
                    scores.append(score)
                    self.train_episode(True)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("/home/yuhao/PycharmProjects/RLTest/a2c/save_graph/keras-a2c-batch-cartpole.png")
                    break

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def memory(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.discount_rewards(self.rewards, done)

        values = self.critic.predict(np.array(self.states))
        values = np.reshape(values, len(values))

        advantages = np.zeros((len(self.rewards), self.action_size))
        for t in range(len(self.rewards)):
            advantages[t][self.actions[t]] = discounted_rewards[t] - values[t]

        self.actor.fit(np.asarray(self.states), advantages, epochs=1)
        self.critic.fit(np.asarray(self.states), discounted_rewards, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

    # def get_action(self, state):
    #     policy = self.actor.predict(np.reshape(state, [1, self.state_size]), batch_size=1)[0]
    #     return np.random.choice(self.action_size, 1, p=policy)[0]

    def get_action(self, state):
        bs = np.reshape(state, [1, self.state_size])
        pred = self.actor.predict(bs, batch_size=1)
        policy = pred.flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    env.close()

    global_agent = A3CAgent(state_size, action_size, env_name)
    global_agent.train()