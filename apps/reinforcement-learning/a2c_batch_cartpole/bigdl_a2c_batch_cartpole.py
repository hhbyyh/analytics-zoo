import gym
import pylab
import random
from bigdl.models.utils.model_broadcast import broadcast_model
from bigdl.nn.criterion import CategoricalCrossEntropy, MSECriterion
from bigdl.nn.layer import *
from bigdl.optim.optimizer import Optimizer, MaxIteration, MaxEpoch, Adam, SGD
from zoo.common.nncontext import *

from a2c.zoo_a2c_cartpole.A2CAgent import A2C_worker_Agent

EPISODES = 100000


class A2C_CartPole_Driver:
    def __init__(self, env_name, state_size, action_size):
        # get size of state and action
        self.state_size = 4
        self.action_size = 2
        self.value_size = 1
        self.hidden = 24

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.agents = [A2C_worker_Agent(env_name, state_size, action_size)]

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Linear(self.state_size, 24))
        actor.add(ReLU())
        actor.add(Linear(24, self.action_size))
        actor.add(SoftMax())
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Linear(self.state_size, 24))
        critic.add(ReLU())
        critic.add(Linear(24, self.value_size))
        return critic

    def collect_samples(self, sc):
        samples = []
        for i in range(len(self.agents)):
            samples.append(self.agents[i].sample_episodes(self.actor, self.critic, 1))

        return samples


if __name__ == "__main__":

    env_name = 'CartPole-v1'

    # test environment on driver
    env = gym.make(env_name)
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    sparkConf = create_spark_conf().setAppName("ss").setMaster("local[1]")
    sc = getOrCreateSparkContext(conf=sparkConf)
    redire_spark_logs()
    init_engine()
    init_executor_gateway(sc)

    driver = A2C_CartPole_Driver(env_name, state_size, action_size)
    actor_lr = 0.001
    critic_lr = 0.001
    scores, episodes = [], []

    for e in range(EPISODES):
        samplesRDD = driver.collect_samples(sc)
        score_stats = list(map(lambda x: x[2], samplesRDD))
        print("epoch e:", e, "  score avg: ", np.mean(score_stats))

        actor_samples = [item for list in list(map(lambda x: x[0], samplesRDD)) for item in list]
        critic_samples = [item for list in list(map(lambda x: x[1], samplesRDD)) for item in list]
        actor_samples = sc.parallelize(actor_samples)
        critic_samples = sc.parallelize(critic_samples)

        num_records = actor_samples.count()
        batch_size = num_records - num_records % 4
        batch_size = 32 if batch_size > 32 else batch_size

        scores.append(np.mean(score_stats))
        episodes.append(e)
        pylab.plot(episodes, scores, 'b')
        pylab.savefig("/home/yuhao/PycharmProjects/RLTest/a2c/save_graph/bigdl_a2c_batch_cartpole.png")

        actor_optimizer = Optimizer(model=driver.actor,
                                    training_rdd=actor_samples,
                                    criterion=CategoricalCrossEntropy(),
                                    optim_method={driver.actor.name(): SGD(learningrate=actor_lr)},
                                    end_trigger= MaxEpoch(1),
                                    batch_size=batch_size)

        critic_optimizer = Optimizer(model=driver.critic,
                                     training_rdd=critic_samples,
                                     criterion=MSECriterion(),
                                     optim_method={driver.critic.name(): SGD(learningrate=critic_lr)},
                                     end_trigger= MaxEpoch(1),
                                     batch_size=batch_size)

        driver.actor = actor_optimizer.optimize()
        driver.critic = critic_optimizer.optimize()
