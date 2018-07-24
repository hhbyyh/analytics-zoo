import sys
import gym
import pylab
import numpy as np
from zoo.pipeline.api.keras.layers import Dense
from zoo.pipeline.api.keras.models import Sequential
from bigdl.optim.optimizer import SGD
from zoo.common.nncontext import *

EPISODES = 1000

# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.001

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        init='glorot_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        init='glorot_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=SGD(learningrate=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         init='glorot_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         init='glorot_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=SGD(learningrate=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        pred = np.asarray(self.actor.predict(state, distributed=False))
        policy = pred.flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state, distributed=False)[0]
        next_value = self.critic.predict(next_state, distributed=False)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, nb_epoch=1, batch_size=1, distributed=False)
        self.critic.fit(state, target, nb_epoch=1, batch_size=1, distributed=False)


if __name__ == "__main__":

    sparkConf = create_spark_conf().setAppName("ss").setMaster("local[1]")
    sc = getOrCreateSparkContext(conf=sparkConf)
    redire_spark_logs()
    init_engine()
    init_executor_gateway(sc)

    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # every episode, plot the play time
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("/home/yuhao/PycharmProjects/RLTest/a2c/save_graph/zoo-a2c-local-cartpole.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(100, len(scores)):]) > 490:
                    sys.exit()
