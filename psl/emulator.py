"""
Base Classes for dynamic systems emulators

"""
from abc import ABC, abstractmethod
import random
import numpy as np
from scipy.integrate import odeint
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
        self.nsim, self.ninit, self.ts = nsim, ninit, ts
        self.parameters()

    @abstractmethod
    def parameters(self):
        """
        Initialize parameters of the dynamical system
        """
        pass

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


class SSM(EmulatorBase):
    """
    base class state space model
    """

    def parameters(self):
        # steady state values
        self.x_ss = 0
        self.y_ss = 0

    def simulate(self, ninit=None, nsim=None, U=None, D=None, x0=None, **kwargs):
        """
        :param nsim: (int) Number of steps for open loop response
        :param U: (ndarray, shape=(nsim, self.nu)) control signals
        :param D: (ndarray, shape=(nsim, self.nd)) measured disturbance signals
        :param x: (ndarray, shape=(self.nx)) Initial state.
        :return: The response matrices, i.e. X, Y, U, D
        """
        if ninit is None:
            ninit = self.ninit
        if nsim is None:
            nsim = self.nsim
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        if D is None:
            D = self.D[ninit: ninit + nsim, :] if self.D is not None else None
        if U is None:
            U = self.U[ninit: ninit + nsim, :] if self.U is not None else None
        X, Y = [], []
        for k in range(nsim):
            u = U[k, :] if U is not None else None
            d = D[k, :] if D is not None else None
            x, y = self.equations(x, u, d)
            X.append(x + self.x_ss)
            Y.append(y - self.y_ss)
        Xout = np.asarray(X).reshape(nsim, self.nx)
        Yout = np.asarray(Y).reshape(nsim, self.ny)
        Uout = np.asarray(U).reshape(nsim, self.nu) if U is not None else None
        Dout = np.asarray(D).reshape(nsim, self.nd) if D is not None else None
        return {'X': Xout, 'Y': Yout, 'U': Uout, 'D': Dout}


class ODE_Autonomous(EmulatorBase):
    """
    base class autonomous ODE
    """

    def parameters(self):
        pass

    def simulate(self, ninit=None, nsim=None, ts=None, x0=None, **kwargs):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param x0: (float) state initial conditions
        :param x: (ndarray, shape=(self.nx)) states
        :return: The response matrices, i.e. X
        """

        # default simulation setup parameters
        if ninit is None:
            ninit = self.ninit
        if nsim is None:
            nsim = self.nsim
        if ts is None:
            ts = self.ts

        # initial conditions states + uncontrolled inputs
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        # time interval
        t = np.arange(0, nsim+1) * ts + ninit
        X = []
        for N in range(nsim):
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT)
            x = xdot[-1]
            X.append(x)  # updated states trajectories
        Yout = np.asarray(X).reshape(nsim, -1)
        return {'Y': Yout, 'X': np.asarray(X)}


class ODE_NonAutonomous(EmulatorBase, ABC):
    """
    base class autonomous ODE
    """

    def parameters(self):
        pass

    def simulate(self, U=None, ninit=None, nsim=None, ts=None, x0=None, **kwargs):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param x0: (float) state initial conditions
        :param x: (ndarray, shape=(self.nx)) states
        :return: X, Y, U, D
        """

        # default simulation setup parameters
        if ninit is None:
            ninit = self.ninit
        if nsim is None:
            nsim = self.nsim
        if ts is None:
            ts = self.ts
        if U is None:
            U = self.U

        # initial conditions states + uncontrolled inputs
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        # time interval
        t = np.arange(0, nsim+1) * ts + ninit
        X = []
        N = 0
        for u in U:
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT, args=(u,))
            x = xdot[-1]
            X.append(x)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        Yout = np.asarray(X).reshape(nsim, -1)
        Uout = np.asarray(U).reshape(nsim, -1)
        return {'Y': Yout, 'U': Uout, 'X': np.asarray(X)}


class GymWrapper(EmulatorBase):
    """
    wrapper for OpenAI gym environments
    https://gym.openai.com/read-only.html
    https://github.com/openai/gym
    # TODO: include visualization option for the trajectories or render of OpenAI gym
    """
    def __init__(self, nsim=1000, ninit=0, system='Pendulum-v0', seed=59):
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
        self.U = Steps(nx=1, nsim=self.nsim, values=None,
                       randsteps=int(np.ceil(self.nsim/40)), xmax=0.5, xmin=-0.5)
        if type(self.action_sample) == int:
            self.U = self.U.astype(int)

    def equations(self, x, u):
        if type(self.action_sample) == int:
            u = u.item()
        self.env.state = x
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
