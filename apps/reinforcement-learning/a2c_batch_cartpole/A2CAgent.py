import numpy as np
from bigdl.util.common import JTensor, Sample
import gym


# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2C_worker_Agent(object):

    def __init__(self, env_name, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.env_name = env_name
        self.value_size = 1
        self.discount_factor = 0.99

        self.env = gym.make(env_name)
        self.last_obv = self.env.reset()

    def get_action(self, state, local_actor):
        state = np.expand_dims(state, axis=0) #np.reshape(state, [1, self.state_size])
        policy = local_actor.predict(state).flatten()
        action =  np.random.choice(self.action_size, 1, p=policy)[0]
        assert action == 0 or action == 1
        return action

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def sample_episodes(self, local_actor, local_critic, num_episodes):
        env = gym.make(self.env_name)
        episode = 0
        actor_samples = []
        critic_samples = []
        scores = []
        while episode < num_episodes:
            state = env.reset()
            score = 0
            states = []
            rewards = []
            actions = []
            while True:
                action = self.get_action(state, local_actor)
                next_state, reward, done, info = env.step(action)
                score += reward

                states.append(state)
                rewards.append(reward)
                actions.append(action)

                state = next_state
                if done:
                    episode += 1
                    scores.append(score)
                    states.append(next_state)
                    break

            discounted_rewards = self.discount_rewards(rewards)
            values = local_critic.predict(np.array(states))
            advantages = np.zeros((len(rewards), self.action_size))

            for t in range(0, len(rewards)):
                advantages[t][actions[t]] = discounted_rewards[t] - values[t]
                actor_samples.append(Sample.from_ndarray(states[t].squeeze(), advantages[t].squeeze()))
                critic_samples.append(Sample.from_ndarray(states[t].squeeze(), discounted_rewards[t].squeeze()))

        return [actor_samples, critic_samples, scores]

        # self.score = 7
    # def sample_steps(self, local_actor, local_critic, num_steps):
    #     actor_samples = []
    #     critic_samples = []
    #     step = 0
    #
    #     state = self.last_obv
    #     states = []
    #     rewards = []
    #     actions = []
    #
    #     while step < num_steps:
    #         step += 1
    #         action = self.get_action(state, local_actor)
    #         next_state, reward, done, info = self.env.step(action)
    #         self.score += reward
    #
    #         states.append(state)
    #         rewards.append(reward)
    #         actions.append(action)
    #
    #         state = next_state
    #         self.last_obv = next_state
    #         print("current score: " * 10, self.score)
    #         if done:
    #             self.score = 0
    #             self.env = gym.make(self.env_name)
    #             self.last_obv = self.env.reset()
    #             break
    #
    #     discounted_rewards = self.discount_rewards(states, rewards, True, local_critic)
    #     values = local_critic.predict(np.array(states))
    #     advantages = np.zeros((len(rewards), self.action_size))
    #
    #     for t in range(0, len(rewards)):
    #         advantages[t][actions[t]] = discounted_rewards[t] - values[t]
    #         actor_samples.append(Sample.from_ndarray(states[t].squeeze(), advantages[t].squeeze()))
    #         critic_samples.append(Sample.from_ndarray(states[t].squeeze(), discounted_rewards[t].squeeze()))
    #
    #     return [actor_samples, critic_samples, [self.score]]
    #
    #
    #
    #
    #
    #     # def memory(self, state, action, reward, next_state):
    #     #     self.states.append(state)
    #     #     act = np.zeros(self.action_size)
    #     #     act[action] = 1
    #     #     self.actions.append(act)
    #     #     self.rewards.append(reward)
    #     #     self.next_states.append(next_state)
    #
    #     # done = False
    #     # state = self.env.reset()
    #     # state = np.reshape(state, [1, self.state_size])
    #     # memory = []
    #     # score = 0
    #     #
    #     # while not done:
    #     #     action = self.get_action(state, local_actor)
    #     #     next_state, reward, done, info = self.env.step(action)
    #     #     next_state = np.reshape(next_state, [1, self.state_size])
    #     #
    #     #     # if an action make the episode end, then gives penalty of -100
    #     #     reward = reward if not done or score == 499 else -100
    #     #     score += reward
    #     #
    #     #     target = np.zeros((1, self.value_size))
    #     #     advantages = np.zeros((1, self.action_size))
    #     #
    #     #     value = local_critic.predict(state)[0]
    #     #     next_value = local_critic.predict(next_state)[0]
    #     #
    #     #     if done:
    #     #         advantages[0][action] = reward - value
    #     #         target[0][0] = reward
    #     #         score = score if score == 500.0 else score + 100
    #     #     else:
    #     #         advantages[0][action] = reward + self.discount_factor * (next_value) - value
    #     #         target[0][0] = reward + self.discount_factor * next_value
    #     #
    #     #     actor_sample = Sample.from_ndarray(state.squeeze(), advantages.squeeze())
    #     #     critic_sample = Sample.from_ndarray(state.squeeze(), target.squeeze())
    #     #     memory.append([actor_sample, critic_sample, score])
    #     #
    #     #     state = next_state
    #     #
    #     # return memory
    #
    # # def sample_steps(self, local_actor, local_critic, num_steps):
    # #     done = False
    # #     state = self.last_obv
    # #     samples = []
    # #     score = 0
    # #
    # #     while len(samples) < num_steps:
    # #         while not done:
    # #             action = self.get_action(state, local_actor)
    # #             next_state, reward, done, info = self.env.step(action)
    # #             next_state = np.reshape(next_state, [1, self.state_size])
    # #
    # #             # if an action make the episode end, then gives penalty of -100
    # #             reward = reward if not done or score == 499 else -100
    # #             score += reward
    # #
    # #             target = np.zeros((1, self.value_size))
    # #             advantages = np.zeros((1, self.action_size))
    # #
    # #             value = local_critic.predict(state)[0]
    # #             next_value = local_critic.predict(next_state)[0]
    # #
    # #             if done:
    # #                 advantages[0][action] = reward - value
    # #                 target[0][0] = reward
    # #                 score = score if score == 500.0 else score + 100
    # #             else:
    # #                 advantages[0][action] = reward + self.discount_factor * (next_value) - value
    # #                 target[0][0] = reward + self.discount_factor * next_value
    # #
    # #             actor_sample = Sample.from_ndarray(state.squeeze(), advantages.squeeze())
    # #             critic_sample = Sample.from_ndarray(state.squeeze(), target.squeeze())
    # #             samples.append([actor_sample, critic_sample, score])
    # #
    # #             state = next_state
    # #
    # #
    # #
    # #     self.last_obv = state
    # #
    # #     return samples
    #
    #
    # #
    # #
    # # # using the output of policy network, pick action stochastically
    # # def get_action(self, state):
    # #     policy = self.actor.predict(state, distributed=False).flatten()
    # #     return np.random.choice(self.action_size, 1, p=policy)[0]
    # #
    # # def append_sample(self, state, action, reward, next_state, done):
    # #     self.memory.append((state, action, reward, next_state, done))
    # #     if self.epsilon > self.epsilon_min:
    # #         self.epsilon *= self.epsilon_decay
    # #
    # # # update policy network every episode
    # # def train_model(self, state, action, reward, next_state, done):
    # #     target = np.zeros((1, self.value_size))
    # #     advantages = np.zeros((1, self.action_size))
    # #
    # #     value = self.critic.predict(state, distributed=False)[0]
    # #     next_value = self.critic.predict(next_state, distributed=False)[0]
    # #
    # #     if done:
    # #         advantages[0][action] = reward - value
    # #         target[0][0] = reward
    # #     else:
    # #         advantages[0][action] = reward + self.discount_factor * (next_value) - value
    # #         target[0][0] = reward + self.discount_factor * next_value
    # #
    # #     self.actor.fit(state, advantages, nb_epoch=1, batch_size=1, distributed=False)
    # #     self.critic.fit(state, target, nb_epoch=1, batch_size=1, distributed=False)
    # #
    # # # def getSample(self):
    #
