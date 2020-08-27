"""
Wrapper for OpenAI gym environments

# TODO: include visualization option for the trajectories or render of OpenAI gym
# TODO: double check dimensions of x for OpenAI gym models

"""

import numpy as np
import gym
import random

# local imports
from psl.emulator import EmulatorBase
from psl.perturb import SplineSignal
from psl.perturb import Steps


class GymWrapper(EmulatorBase):
    """
    wrapper for OpenAI gym environments
    https://gym.openai.com/read-only.html
    https://github.com/openai/gym
    """
    def __init__(self, nsim=1000, ninit=0, system='Pendulum-v0'):
        super().__init__(nsim=nsim, ninit=ninit)
        self.system = system

    def parameters(self, system='Pendulum-v0'):
        self.system = system
        self.env = gym.make(self.system)
        self.env.reset()  # to reset the environment state
        self.x0 = self.env.state
        self.nx = self.x0.shape[0]
        self.action_sample = self.env.action_space.sample()
        self.nu = np.asarray([self.action_sample]).shape[0]
        #     default simulation setup
        self.U = np.zeros([(self.nsim - 1), self.nu])
        self.U[int(self.nsim/8)] = 0.1
        # randomIndList =[]
        # for i in range(0, 100):
        #     # any random numbers from 0 to 1000
        #     ind = random.randint(0, self.nsim-1)
        #     self.U[ind] = random.uniform(-1, 1)
        self.U = Steps(nx=1, nsim=self.nsim, values=None,
                       randsteps=int(np.ceil(self.nsim/40)), xmax=0.5, xmin=-0.5)
        # self.U = SplineSignal(nsim=self.nsim, values=None, xmin=-2.0, xmax=2.0)
        # print(self.U.shape)
        if type(self.action_sample) == int:
            self.U = self.U.astype(int)

    def equations(self, x, u):
        if type(self.action_sample) == int:
            u = u.item()
        self.env.state = x
        print(u)
        x, reward, done, info = self.env.step(u)
        return x, reward

    def simulate(self, nsim=None, U=None, x0=None, **kwargs):
        """
        :param nsim: (int) Number of steps for open loop response
        :param U: (ndarray, shape=(self.nu)) control actions
        :param x0: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response trajectories,  X
        """
        if nsim is None:
            nsim = self.nsim
        if U is None:
            U = self.U
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0

        X, Reward = [], []
        N = 0
        for u in U:
            x, reward = self.equations(x, u)
            X.append(x)  # updated states trajectories
            Reward.append(reward)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        Xout = np.asarray(X)
        Yout = np.asarray(Reward).reshape(-1, 1)
        Uout = np.asarray(U)
        return {'X': Xout, 'Y': Yout, 'U': Uout}
