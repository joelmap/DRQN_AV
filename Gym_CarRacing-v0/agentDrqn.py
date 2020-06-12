from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam

import pylab
from skimage.color import rgb2gray

from collections import deque

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144]).reshape(96, 96, 1)

class DRQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)
        
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        
        model.add(Conv2D(32, 8, strides=(4, 4), activation='relu', padding='valid', input_shape=(96, 96, 1)))
        model.add(Conv2D(64, 4, strides=(2, 2), activation='relu', padding='valid'))
        model.add(Conv2D(64, 3, strides=(1, 1), activation='relu', padding='valid'))
        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(512, activation='tanh', recurrent_activation='sigmoid'))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(3))

        model.summary()
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = np.expand_dims(state, axis = 0)
        actions = self.model.predict(state)[0]
        return actions

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        state_size = (96, 96, 1)
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)

        state = np.zeros(((batch_size,) + state_size)) #32x3
        new_state = np.zeros(((batch_size,) + state_size))
        for sample in samples:
            state[:,:,:,:], action, reward, new_state[:,:,:,:], done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][1] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][1] = reward + Q_future * self.gamma

            train = self.model.fit(state, target, epochs=10, verbose=0, validation_split=0.2)

            pylab.gcf().clear()
            pylab.plot(train.history['acc'], 'b', label='training')
            pylab.plot(train.history['val_acc'], 'orange', label='validation')
            pylab.title('Accuracy x epochs')
            pylab.xlabel('Epochs')
            pylab.ylabel('Accuracy')
            pylab.legend(loc='upper right')
            pylab.savefig('acc' + '.png')

            pylab.gcf().clear()
            pylab.plot(train.history['loss'], 'b', label='training')
            pylab.plot(train.history['val_loss'], 'orange', label='validation')
            pylab.title('Loss x epochs')
            pylab.xlabel('Epochs')
            pylab.ylabel('Loss')
            pylab.legend(loc='upper right')
            pylab.savefig('loss' + '.png')

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

if __name__ == "__main__":
    env     = gym.make("CarRacing-v0")
    env.seed(0)
    gamma   = 0.9
    epsilon = .95

    trials  = 100 

    dqn_agent = DRQN(env=env)

    total_rewards, episodes, filtered_scores = [], [], []

    for trial in range(trials):
        cur_state = env.reset()
        cur_state = rgb2gray(cur_state)
        total_reward = 0
        done = False
        steps = 0
        while not done:
            env.render()
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            new_state = rgb2gray(new_state)
            
            steps += 1

            print(f"Step#:{steps}  Reward:{reward}  Done:{done}")
            
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            total_reward += reward
            cur_state = new_state
            
            if done:
                if len(filtered_scores) != 0: # if list is not empty
                    filtered_scores.append(0.98*filtered_scores[-1] + 0.02*total_reward)
                else: # if list is empty
                    filtered_scores.append(total_reward)
                total_rewards.append(total_reward)
                episodes.append(trial)
                pylab.gcf().clear()
                pylab.plot(episodes, total_rewards, 'b', label='Total rewards')
                pylab.plot(episodes, filtered_scores, 'orange', label='Rewards average')
                pylab.title('Total rewards x episodes')
                pylab.xlabel('Episode')
                pylab.ylabel('Total rewards')
                pylab.legend(loc='upper right')
                pylab.savefig('testeDRQN' + '.png')

                dqn_agent.save_model("trial-{}.model".format(trial))

                break

        print("episode: {:3}   total reward: {:8.6}   memory length: {:4}   epsilon {:.3}"
                            .format(trial, float(total_reward), len(dqn_agent.memory), float(dqn_agent.epsilon)))

        # save the model every N episodes
        if trial % 100 == 0: #100
            # if not TEST:
            print("Now we save model")
            dqn_agent.save_model("success.model")
    env.close()