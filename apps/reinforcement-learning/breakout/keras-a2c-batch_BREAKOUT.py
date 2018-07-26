
import numpy as np
import gym
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
import pylab
from skimage.color import rgb2gray
from skimage.transform import resize


class GlobalAgent:
    def __init__(self, state_size, action_size, env_name):
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # get gym environment name
        self.env_name = env_name

        # these are hyper parameters for the A3C
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        self.discount_factor = .99
        self.hidden1, self.hidden2 = 24, 24
        self.threads = 1

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()


    def build_model(self):
        input = Input(shape=self.state_size)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)
        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.actor_lr))
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return actor, critic


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

    def pre_processing(self, next_observe, observe):
        processed_observe = np.maximum(next_observe, observe)
        processed_observe = np.uint8(resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
        return processed_observe

    # Thread interactive with environment
    def run(self):
        episode = 0
        env = gym.make(env_name)


        episodes = []
        scores = []
        EPISODES = 200000
        while episode < EPISODES:
            done = False
            dead = False
            step = 0
            # 1 episode = 5 lives
            score, start_life = 0, 5
            observe = env.reset()
            next_observe = observe

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(np.random.randint(1, 30)):
                observe = next_observe
                next_observe, _, _, _ = env.step(1)

            # At start of episode, there is no preceding frame. So just copy initial states to make history
            state = self.pre_processing(next_observe, observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                step += 1
                observe = next_observe
                # get action for the current history and go one step in environment
                action, policy = self.get_action(history)
                # change action to real_action
                if action == 0: real_action = 1
                elif action == 1: real_action = 2
                else: real_action = 3

                if dead:
                    action = 0
                    real_action = 1
                    dead = False

                next_observe, reward, done, info = env.step(real_action)
                # pre-process the observation --> history
                next_state = self.pre_processing(next_observe, observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                # if the ball is fall, then the agent is dead --> episode is not over
                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                # save the sample <s, a, r, s'> to the replay memory
                self.memory(history, action, reward)

                # if agent is dead, then reset the history
                if dead:
                    history = np.stack((next_state, next_state, next_state, next_state), axis=2)
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                # if done, plot the score over episodes
                if done:
                    episode += 1
                    print("episode:", episode, "  score:", score, "  step:", step)
                    episodes.append(episode)
                    scores.append(score)
                    self.train_episode()
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("/home/yuhao/PycharmProjects/RLTest/a2c/save_graph/keras-a2c-batch-breakout.png")


    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
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
    def train_episode(self):
        discounted_rewards = self.discount_rewards(self.rewards)

        states = np.zeros((len(self.states), 84, 84, 4))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states / 255.)

        values = self.critic.predict(np.array(states))
        values = np.reshape(values, len(values))

        advantages = np.zeros((len(self.rewards), self.action_size))
        for t in range(len(self.rewards)):
            advantages[t][self.actions[t]] = discounted_rewards[t] - values[t]

        self.actor.fit(np.asarray(states), advantages, epochs=1)
        self.critic.fit(np.asarray(states), discounted_rewards, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy


if __name__ == "__main__":
    env_name = "BreakoutDeterministic-v4"
    env = gym.make(env_name)

    state_size = (84, 84, 4)
    action_size = 3

    env.close()

    global_agent = GlobalAgent(state_size, action_size, env_name)
    global_agent.train()