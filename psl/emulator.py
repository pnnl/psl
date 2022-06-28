"""
Base Classes for dynamic systems emulators

"""
from abc import ABC, abstractmethod
import random
import numpy as np
import gym

from psl.perturb import Steps


class EmulatorBase(ABC):
    """
    base class of the emulator
    """
    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.nsim, self.ninit, self.ts = nsim, ninit, ts
        self.x0 = 0.

    @abstractmethod
    def equations(self, **kwargs):
        """
        Define equations defining the dynamical system
        """
        pass

    @abstractmethod
    def simulate(self, **kwargs):
        """
        N-step forward simulation of the dynamical system
        """
        pass


class GymWrapper(EmulatorBase):
    """
    Wrapper for OpenAI gym environments

        + https://gym.openai.com/read-only.html
        + https://github.com/openai/gym

    """
    envs = ["Pendulum-v1", "CartPole-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0"]
    
    def __init__(self, nsim=1000, ninit=0, system='Pendulum-v1', seed=59):
        super().__init__(nsim=nsim, ninit=ninit, seed=seed)
        self.system = system
        self.env = gym.make(self.system)
        self.env.reset()
        self.x0 = self.env.state
        self.nx = self.x0.shape[0]
        self.action_sample = self.env.action_space.sample()
        self.nu = np.asarray([self.action_sample]).shape[0]
        self.U = Steps(nx=1, nsim=self.nsim, values=None,
                       randsteps=int(np.ceil(self.nsim/40)), xmax=0.5, xmin=-0.5)
        if type(self.action_sample) == int:
            self.U = self.U.astype(int)

    def equations(self, x, u):
        if type(self.action_sample) == int:
            u = u.item()
        self.env.state = x
        x, reward, done, info = self.env.step(u)
        return x, reward, done, info

    def simulate(self, nsim=None, U=None, x0=None):
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

        X, Reward = [x], [0.]
        N = 0
        for u in U:
            x, reward, done, info = self.equations(x, u)
            X.append(x)
            Reward.append(reward)
            N += 1
            if N == nsim-1:
                break
        Xout = np.asarray(X)
        Yout = np.asarray(Reward).reshape(-1, 1)
        Uout = np.asarray(U)
        return {'X': Xout, 'Y': Yout, 'U': Uout}

systems = {k: GymWrapper for k in GymWrapper.envs}