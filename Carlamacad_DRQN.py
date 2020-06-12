#!/usr/bin/env python
from __future__ import print_function

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.python.client import device_lib

from collections import deque

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    LSTM,
    TimeDistributed,
    Dropout,
    Reshape,
    MaxPooling3D,
    MaxPooling2D,
    Conv2D,
    ConvLSTM2D,
    BatchNormalization,
    Conv3D,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

# Multilayer Perceptron
from tensorflow.keras.utils import plot_model
import skimage as skimage
from skimage.transform import resize
from skimage.color import rgb2gray
from tensorflow.python.keras import backend as K
import numpy as np
import random
import gym
import macad_gym

import sys
import pylab

from pandas import DataFrame
from datetime import datetime
from pytz import timezone
import time
import csv
from tensorflow.keras.callbacks import TensorBoard

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == "GPU"]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144]).reshape(84, 84, 1)


episode = []
step = []
TEST = False


class experience_buffer:
    def __init__(self, buffer_size=10000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0 : (1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) - trace_length)
            sampledTraces.append(episode[point : point + 5])
        sampledTraces = np.array(sampledTraces)
        return sampledTraces


# DRQN agent
class DRQNAgent:
    def __init__(self, state_size, action_size, actor_configs):
        # Create replay memory using experience_buffer
        self.memory = experience_buffer()

        self.actor_configs = actor_configs
        self.action_idx = {}

        # Define size of states and actions
        self.state_size = state_size
        self.action_size = action_size

        self.save_loc = "./DRQN_model"

        # DRQN hyperparameters
        self.gamma = 0.99
        self.learning_rate = 0.01  #0.00025  #0.001
        self.epsilon = 1.0
        self.final_epsilon = 0.01 
        self.epsilon_decay = 0.95
        self.tau = 0.125

        # Create main model and target model
        self.model = self.build_model_drqn(self.state_size, self.action_size)
        self.target_model = self.build_model_drqn(self.state_size, self.action_size)

    def build_model_drqn(self, input_shape, action_size):

        model = Sequential()
        model.add(
            TimeDistributed(
                Conv2D(32, 8, strides=(4, 4), activation="relu"),
                input_shape=(4, 84, 84, 1),
            )
        )
        model.add(TimeDistributed(Conv2D(64, 4, strides=(2, 2), activation="relu")))
        model.add(
            TimeDistributed(
                Conv2D(64, 3, strides=(1, 1), activation="relu", padding="valid")
            )
        )
        model.add(MaxPooling3D(pool_size=(1, 2, 2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(512, activation="tanh"))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(self.action_size, activation="softmax"))

        model.summary()

        model.compile(
            loss="mse",
            optimizer=Adam(lr=self.learning_rate),
            metrics=["accuracy"],  # tf.keras.metrics.Precision()
        )

        # plot_model(model, to_file="model.png")

        return model

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        for actor_id in self.actor_configs.keys():
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.final_epsilon, self.epsilon)

            if np.random.rand() < self.epsilon:
                self.action_idx[actor_id] = random.randrange(self.action_size)
            else:
                if type(state) is dict:
                    state = state.get("car1")
                state = np.expand_dims(state, axis=0)
                state = np.resize(state, (4, 84, 84, 1))
                state = np.expand_dims(state, axis=0)
                self.action_idx[actor_id] = np.argmax(self.model.predict(state))
        return self.action_idx

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):

        if len(self.memory.buffer) < batch_size:
            return

        mini_batch = self.memory.sample(batch_size, trace_length)

        state = np.zeros(((batch_size,) + self.state_size))  # 32x4x84x84x1
        new_state = np.zeros(((batch_size,) + self.state_size))

        # like Q Learning, get maximum Q value at state
        # But from target model
        for i in mini_batch:
            state[:, :, :, :, :], action, reward, new_state[:, :, :, :, :], done = i
            target = self.target_model.predict(state)
            if done:
                target[0][action["car1"]] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action["car1"]] = reward + Q_future * self.gamma

            historico = self.model.fit(
                state, target, epochs=15, verbose=0, validation_split=0.2
            )

            pylab.gcf().clear()
            pylab.plot(historico.history["acc"], "b", label="training")
            pylab.plot(historico.history["val_acc"], "orange", label="validation")
            pylab.title("Accuracy x epoch")
            pylab.xlabel("Epochs")
            pylab.ylabel("Accuracy")
            pylab.legend(loc="upper right")
            pylab.savefig("acc" + ".png")

            pylab.gcf().clear()
            pylab.plot(historico.history["loss"], "b", label="training")
            pylab.plot(historico.history["val_loss"], "orange", label="validation")
            pylab.title("Loss x epoch")
            pylab.xlabel("Epochs")
            pylab.ylabel("Loss")
            pylab.legend(loc="upper right")
            pylab.savefig("loss" + ".png")

    def update_target_model(self):
        # After some time interval update the target model to be same with model
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (
                1 - self.tau
            )
        self.target_model.set_weights(target_weights)

    # save the model which is under training
    def save_model(self, fn):
        self.model.save(fn)


if __name__ == "__main__":

    env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")
    env_config = env.configs["env"]
    actor_configs = env.configs["actors"]

    img_rows, img_cols = 84, 84
    img_channels = 1  # Color channel

    batch_size = 32  # How many experience traces to use for each training step.
    trace_length = 4  # How long each experience trace will be when training

    state_size = (trace_length, img_rows, img_cols, img_channels)
    action_size = 9

    agent = DRQNAgent(state_size, action_size, actor_configs)

    total_rewards, filtered_scores = [], []

    time1 = timezone("Canada/Eastern")

    for e in range(201):

        cur_state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        cur_state["car1"] = rgb2gray(cur_state["car1"])
        cur_stateC = cur_state["car1"]

        dt_date = datetime.now(time1)

        if TEST:
            agent.epsilon = 0.0

        while not done:
            # get action for the current state and go one step in environment
            general_action = agent.get_action(cur_state)
            obs = env.step(general_action)
            key = ("car1",)
            value = general_action["car1"]
            action = dict.fromkeys(key, value)
            next_state = obs[0]["car1"]
            reward = obs[3]["car1"]["reward"]
            done = obs[2]["car1"]

            next_state = rgb2gray(next_state)
            steps += 1
            dt_date_step = datetime.now(time1)
            time_step = dt_date_step.strftime("%H:%M:%S")
            print(
                f"Episode#:{e} Step#:{steps}  Reward:{reward}  Done:{done} Step_end_time:{time_step} memory length: {len(agent.memory.buffer)}"
            )
            if not TEST:
                # save the sample <state, action, reward, next_state> to the replay memory
                agent.memory.add(cur_stateC, action, reward, next_state, done)
                # every time step do the training
                agent.train_replay()
                agent.update_target_model()

            reward += reward
            total_reward = reward
            cur_stateC = next_state

            if done:
                # resetting all actors done state to true
                # since we are getting/setting only the car1 done state, not setting
                # done for the other actors will make them not being created on
                # new episodes
                for actor_key in env._done_dict:
                    env._done_dict[actor_key] = True

                if len(filtered_scores) != 0:  # if list is not empty
                    filtered_scores.append(
                        0.98 * filtered_scores[-1] + 0.02 * total_reward
                    )
                else:  # if list is empty
                    filtered_scores.append(total_reward)
                total_rewards.append(total_reward)
                episode.append(e)
                pylab.gcf().clear()
                pylab.plot(episode, total_rewards, "b", label="Total rewards")
                pylab.plot(episode, filtered_scores, "orange", label="Rewards average")
                pylab.title("Rewards per episode")
                pylab.xlabel("Episodes")
                pylab.ylabel("Total rewards")
                pylab.legend(loc="upper right")
                pylab.savefig(agent.save_loc + ".png")

                print("Completed in {} trials".format(steps))
                agent.save_model("successCurva-{}.model".format(steps))

        print(
            "episode: {:3}   total reward: {:8.6}   memory length: {:4}   epsilon: {:.3} date: {}".format(
                e,
                float(total_reward),
                len(agent.memory.buffer),
                float(agent.epsilon),
                dt_date,
            )
        )

        step.append(steps)
        pylab.gcf().clear()
        pylab.plot(episode, step, "b")
        pylab.title("Steps per episode")
        pylab.xlabel("Episodes")
        pylab.ylabel("Steps")
        pylab.savefig("steps" + ".png")

        # save the model every N episodes
        if e % 100 == 0:  # 100
            if not TEST:
                print("Now we save model")
                agent.save_model("trialCurva-{}.model".format(e))
