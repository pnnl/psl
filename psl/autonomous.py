"""
Nonlinear ODEs. Wrapper for emulator dynamical models

    + Internal Emulators - in house ground truth equations
    + External Emulators - third party models

References:

    + https://en.wikipedia.org/wiki/List_of_nonlinear_ordinary_differential_equations
    + https://en.wikipedia.org/wiki/List_of_dynamical_systems_and_differential_equations_topics

"""
import numpy as np
import inspect, sys

from psl.emulator import EmulatorBase
from scipy.integrate import odeint


class ODE_Autonomous(EmulatorBase):
    """
    base class autonomous ODE
    """

    def simulate(self, ninit=None, nsim=None, ts=None, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param x0: (float) state initial conditions
        :return: The response matrices, i.e. X
        """

        if ninit is None:
            ninit = self.ninit
        if nsim is None:
            nsim = self.nsim
        if ts is None:
            ts = self.ts

        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        t = np.arange(0, nsim+1) * ts + ninit
        X = [x]
        for N in range(nsim-1):
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT)
            x = xdot[-1]
            X.append(x)
        Yout = np.asarray(X).reshape(nsim, -1)
        return {'Y': Yout, 'X': np.asarray(X)}


class UniversalOscillator(ODE_Autonomous):
    """
    Harmonic oscillator

    + https://en.wikipedia.org/wiki/Harmonic_oscillator
    + https://sam-dolan.staff.shef.ac.uk/mas212/notebooks/ODE_Example.html
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.mu = 2
        self.omega = 1
        self.x0 = [1.0, 0.0]
        self.nx = 2

    def equations(self, x, t):
        dx1 = x[1]
        dx2 = -2*self.mu*x[1] - x[0] + np.cos(self.omega*t)
        dx = [dx1, dx2]
        return dx


class Pendulum(ODE_Autonomous):
    """
    Simple pendulum.

    + https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.g = 9.81
        self.f = 3.
        self.nx = 2
        self.x0 = [0., 1.]

    def equations(self, x, t):
        theta = x[0]
        omega = x[1]
        return [omega, -self.f*omega - self.g*np.sin(theta)]


class DoublePendulum(ODE_Autonomous):
    """
    Double Pendulum
    https://scipython.com/blog/the-double-pendulum/
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.L1 = 1
        self.L2 = 1
        self.m1 = 1
        self.m2 = 1
        self.g = 9.81
        self.x0 = np.array([3 * np.pi / 7, 0, 3 * np.pi / 4, 0])
        self.nx = 4

    def equations(self, x, t):
        theta1 = x[0]
        z1 = x[1]
        theta2 = x[2]
        z2 = x[3]
        c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)
        dx1 = z1
        dx2 = (self.m2 * self.g * np.sin(theta2) * c - self.m2 * s * (self.L1 * z1 ** 2 * c + self.L2 * z2 ** 2) -
                 (self.m1 + self.m2) * self.g * np.sin(theta1)) / self.L1 / (self.m1 + self.m2 * s ** 2)
        dx3 = z2
        dx4 = ((self.m1 + self.m2) * (self.L1 * z1 ** 2 * s - self.g * np.sin(theta2) + self.g * np.sin(theta1) * c) +
                 self.m2 * self.L2 * z2 ** 2 * s * c) / self.L2 / (self.m1 + self.m2 * s ** 2)
        dx = [dx1, dx2, dx3, dx4]
        return dx


class Lorenz96(ODE_Autonomous):
    """
    Lorenz 96 model

    + https://en.wikipedia.org/wiki/Lorenz_96_model
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.N = 36  # Number of variables
        self.F = 8  # Forcing
        self.x0 = self.F*np.ones(self.N)
        self.x0[19] += 0.01  # Add small perturbation to random variable
        self.nx = self.N

    def equations(self, x, t):
        dx = np.zeros(self.N)
        # First the 3 edge cases: i=1,2,N
        dx[0] = (x[1] - x[self.N - 2]) * x[self.N - 1] - x[0]
        dx[1] = (x[2] - x[self.N - 1]) * x[0] - x[1]
        dx[self.N - 1] = (x[0] - x[self.N - 3]) * x[self.N - 2] - x[self.N - 1]
        # Then the general case
        for i in range(2, self.N - 1):
            dx[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]
        # Add the forcing term
        dx = dx + self.F
        return dx


class LorenzSystem(ODE_Autonomous):
    """
    Lorenz System

    + https://en.wikipedia.org/wiki/Lorenz_system#Analysis
    + https://ipywidgets.readthedocs.io/en/stable/examples/Lorenz%20Differential%20Equations.html
    + https://scipython.com/blog/the-lorenz-attractor/
    + https://matplotlib.org/3.1.0/gallery/mplot3d/lorenz_attractor.html
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.x0 = [1.0, 1.0, 1.0]
        self.nx = 3

    def equations(self, x, t):
        dx1 = self.sigma*(x[1] - x[0])
        dx2 = x[0]*(self.rho - x[2]) - x[1]
        dx3 = x[0]*x[1] - self.beta*x[2]
        dx = [dx1, dx2, dx3]
        return dx


class VanDerPol(ODE_Autonomous):
    """
    Van der Pol oscillator

    + https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    + http://kitchingroup.cheme.cmu.edu/blog/2013/02/02/Solving-a-second-order-ode/
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.mu = 1.0
        self.x0 = [1, 2]
        self.nx = 2

    def equations(self, x, t):
        dx1 = self.mu*(x[0] - 1./3.*x[0]**3 - x[1])
        dx2= x[0]/self.mu
        dx = [dx1, dx2]
        return dx


class ThomasAttractor(ODE_Autonomous):
    """
    Thomas' cyclically symmetric attractor

    + https://en.wikipedia.org/wiki/Thomas%27_cyclically_symmetric_attractor
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.b = 0.208186
        self.x0 = [1, -1, 1]
        self.nx = 3

    def equations(self, x, t):
        dx1 = np.sin(x[1]) - self.b*x[0]
        dx2 = np.sin(x[2]) - self.b*x[1]
        dx3 = np.sin(x[0]) - self.b*x[2]
        dx = [dx1, dx2, dx3]
        return dx


class RosslerAttractor(ODE_Autonomous):
    """
    Rössler attractor

    + https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.a = 0.2
        self.b = 0.2
        self.c = 5.7
        self.x0 = [0, 0, 0]
        self.nx = 3

    def equations(self, x, t):
        dx1 = - x[1] - x[2]
        dx2 = x[0] + self.a*x[1]
        dx3 = self.b + x[2]*(x[0]-self.c)
        dx = [dx1, dx2, dx3]
        return dx


class LotkaVolterra(ODE_Autonomous):
    """
    Lotka–Volterra equations, also known as the predator–prey equations

    + https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.a = 1.
        self.b = 0.1
        self.c = 1.5
        self.d = 0.75
        self.x0 = [5, 100]
        self.nx = 2

    def equations(self, x, t):
        dx1 = self.a*x[0] - self.b*x[0]*x[1]
        dx2 = -self.c*x[1] + self.d*self.b*x[0]*x[1]
        dx = [dx1, dx2]
        return dx


class Brusselator1D(ODE_Autonomous):
    """
    Brusselator

    + https://en.wikipedia.org/wiki/Brusselator
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.a = 1.0
        self.b = 3.0
        self.x0 = [1.0, 1.0]
        self.nx = 2

    def equations(self, x, t):
        dx1 = self.a + x[1]*x[0]**2 -self.b*x[0] - x[0]
        dx2 = self.b*x[0] - x[1]*x[0]**2
        dx = [dx1, dx2]
        return dx


class ChuaCircuit(ODE_Autonomous):
    """
    Chua's circuit

    + https://en.wikipedia.org/wiki/Chua%27s_circuit
    + https://www.chuacircuits.com/matlabsim.php
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.a = 15.6
        self.b = 28.0
        self.m0 = -1.143
        self.m1 = -0.714
        self.x0 = [0.7, 0.0, 0.0]
        self.nx = 3

    def equations(self, x, t):
        fx = self.m1*x[0] + 0.5*(self.m0 - self.m1)*(np.abs(x[0] + 1) - np.abs(x[0] - 1))
        dx1 = self.a*(x[1] - x[0] - fx)
        dx2 = x[0] - x[1] + x[2]
        dx3 = -self.b*x[1]
        dx = [dx1, dx2, dx3]
        return dx


class Duffing(ODE_Autonomous):
    """
    Duffing equation

    + https://en.wikipedia.org/wiki/Duffing_equation
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.delta = 0.02
        self.alpha = 1
        self.beta = 5
        self.gamma = 8
        self.omega = 0.5
        self.x0 = [1.0, 0.0]
        self.nx = 2

    def equations(self, x, t):
        dx1 = x[1]
        dx2 = - self.delta*x[1] - self.alpha*x[0] - self.beta*x[0]**3 + self.gamma*np.cos(self.omega*t)
        dx = [dx1, dx2]
        return dx


class Autoignition(ODE_Autonomous):
    """
    ODE describing pulsating instability in open-ended combustor.

    + Koch, J., Kurosaka, M., Knowlen, C., Kutz, J.N.,
      "Multiscale physics of rotating detonation waves: Autosolitons and modulational instabilities,"
      Physical Review E, 2021
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.alpha = 0.3
        self.uc = 1.1
        self.s = 1.0
        self.k = 1.0
        self.r = 5.0
        self.q = 6.5
        self.up = 0.55
        self.e = 1.0
        self.x0 = [1.0, 0.7]

    def equations(self, x, t):
        reactionRate = self.k * (1.0 - x[1]) * np.exp((x[0] - self.uc) / self.alpha)
        regenRate = self.s * self.up * x[1] / (1.0 + np.exp(self.r * (x[0] - self.up)))
        dx1 = self.q * reactionRate - self.e * x[0] ** 2
        dx2 = reactionRate - regenRate
        dx = [dx1, dx2]
        return dx

systems = dict(inspect.getmembers(sys.modules[__name__], lambda x: inspect.isclass(x)))
systems = {k: v for k, v in systems.items() if issubclass(v, ODE_Autonomous) and v is not ODE_Autonomous}

