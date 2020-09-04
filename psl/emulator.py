"""
Base Classes for dynamic systems emulators

"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import odeint


class EmulatorBase(ABC):
    """
    base class of the emulator
    """
    def __init__(self, nsim=1001, ninit=0, ts=0.1):
        super().__init__()
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
        Yout = np.asarray(X)
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
        Yout = np.asarray(X)
        Uout = np.asarray(U)
        return {'Y': Yout, 'U': Uout, 'X': np.asarray(X)}

