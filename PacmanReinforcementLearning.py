import tensorflow as tf
import gym
import numpy as np

# Load environment
env = gym.make("MsPacman-ram-v0")

print(env.observation_space)
print(env.action_space)

gamesToPlay = 10

for i in range(gamesToPlay):
    obs = env.reset()
    done = False
    gameRewards = 0

    iterationCounter = 0

    # Every iteration the environment is rendered, a random action is chosen, and we step forward with the
    # chosen action

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        gameRewards += reward
        iterationCounter += 1
        print("Iteration " + str(iterationCounter) + " Rewards:" + str(gameRewards))

    env.close()


# Build policy gradient neural network
class Agent:

    def __init__(self, num_actions, state_size):
        initializer = tf.compat.v1.contrib.layers.xavier_initializer()

        self.input_layer = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, state_size])

        # Neural net starts here

        hidden_layer = tf.compat.v1.layers.dense(self.input_layer, 8, activation=tf.nn.relu,
                                                 kernel_initializer=initializer)
        hidden_layer_2 = tf.compat.v1.layers.dense(hidden_layer, 8, activation=tf.nn.relu,
                                                   kernel_initializer=initializer)

        # Output of neural net
        out = tf.compat.v1.layers.dense(hidden_layer_2, num_actions, activation=None)

        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)

        # Training Procedure
        self.rewards = tf.compat.v1.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.compat.v1.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)

        self.loss = tf.reduce_mean(cross_entropy * self.rewards)

        self.gradients = tf.gradients(self.loss, tf.compat.v1.trainable_variables())

        # Create a placeholder list for gradients
        self.gradients_to_apply = []
        for _, variable in enumerate(tf.compat.v1.trainable_variables()):
            gradient_placeholder = tf.compat.v1.placeholder(tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)

        # Create the operation to update gradients with the gradient's placeholder.
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-2)
        self.update_gradients = optimizer.apply_gradients(
            zip(self.gradients_to_apply, tf.compat.v1.trainable_variables()))


discountRate = 0.95


def discount_normalize_rewards(rewards):
    discounted_rewards = np.zeroes_like(rewards)
    total_rewards = 0

    for j in reversed(range(len(rewards))):
        total_rewards *= (discountRate + rewards[j])
        discounted_rewards[j] = total_rewards

    # For each element in the reward list subtract the mean of the discounted rewards list
    discounted_rewards -= np.mean(discounted_rewards)

    # For each element in the list divide by the standard deviation
    discounted_rewards /= np.std(discounted_rewards)

    # Discount rates closer to 1 value future actions higher while discount rates closer to 0 value recent actions
    # higher.
    return discounted_rewards


tf.compat.v1.reset_default_graph()

numActions = 9
stateSize = 128

episode_batch_size = 5

agent = Agent(numActions, stateSize)

# Sets up the global variables  defined in the agents constructor
init = tf.compat.v1.global_variables_initializer

with tf.compat.v1.Session() as sess:
    sess.run(init)
    totalEpisodeAwards = []

    gradientBuffer = sess.run(tf.compat.v1.trainable_variables)

    for index, gradient in enumerate(gradientBuffer):
        gradientBuffer[index] = gradient * 0
