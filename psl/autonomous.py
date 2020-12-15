"""
wrapper for emulator dynamical models
Internal Emulators - in house ground truth equations
External Emulators - third party models
"""
import numpy as np

"""
Nonlinear ODEs

https://en.wikipedia.org/wiki/List_of_nonlinear_ordinary_differential_equations
https://en.wikipedia.org/wiki/List_of_dynamical_systems_and_differential_equations_topics
"""
from psl.emulator import ODE_Autonomous


class UniversalOscillator(ODE_Autonomous):
    """
    Hharmonic oscillator
    https://en.wikipedia.org/wiki/Harmonic_oscillator
    https://sam-dolan.staff.shef.ac.uk/mas212/notebooks/ODE_Example.html
    """

    def parameters(self):
        super().parameters()
        self.mu = 2
        self.omega = 1
        self.x0 = [1.0, 0.0]
        self.nx = 2

    def equations(self, x, t):
        # Derivatives
        dx1 = x[1]
        dx2 = -2*self.mu*x[1] - x[0] + np.cos(self.omega*t)
        dx = [dx1, dx2]
        return dx


class Lorenz96(ODE_Autonomous):
    """
    Lorenz 96 model
    https://en.wikipedia.org/wiki/Lorenz_96_model
    """

    def parameters(self):
        super().parameters() # inherit parameters of the mothership
        self.N = 36  # Number of variables
        self.F = 8  # Forcing
        self.x0 = self.F*np.ones(self.N)
        self.x0[19] += 0.01  # Add small perturbation to random variable
        self.nx = self.N

    def equations(self, x, t):
        # Compute state derivatives
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
    https://en.wikipedia.org/wiki/Lorenz_system#Analysis
    # https://ipywidgets.readthedocs.io/en/stable/examples/Lorenz%20Differential%20Equations.html
    # https://scipython.com/blog/the-lorenz-attractor/
    # https://matplotlib.org/3.1.0/gallery/mplot3d/lorenz_attractor.html
    """

    def parameters(self):
        super().parameters()
        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.x0 = [1.0, 1.0, 1.0]
        self.nx = 3

    def equations(self, x, t):
        # Derivatives
        dx1 = self.sigma*(x[1] - x[0])
        dx2 = x[0]*(self.rho - x[2]) - x[1]
        dx3 = x[0]*x[1] - self.beta*x[2]
        dx = [dx1, dx2, dx3]
        return dx


class VanDerPol(ODE_Autonomous):
    """
    Van der Pol oscillator
    https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    http://kitchingroup.cheme.cmu.edu/blog/2013/02/02/Solving-a-second-order-ode/
    """

    def parameters(self):
        super().parameters()
        self.mu = 1.0
        self.x0 = [1, 2]
        self.nx = 2

    def equations(self, x, t):
        # Derivatives
        dx1 = self.mu*(x[0] - 1./3.*x[0]**3 - x[1])
        dx2= x[0]/self.mu
        dx = [dx1, dx2]
        return dx


class ThomasAttractor(ODE_Autonomous):
    """
    Thomas' cyclically symmetric attractor
    https://en.wikipedia.org/wiki/Thomas%27_cyclically_symmetric_attractor
    """

    def parameters(self):
        super().parameters()
        self.b = 0.208186
        self.x0 = [1, -1, 1]
        self.nx = 3

    def equations(self, x, t):
        # Derivatives
        dx1 = np.sin(x[1]) - self.b*x[0]
        dx2 = np.sin(x[2]) - self.b*x[1]
        dx3 = np.sin(x[0]) - self.b*x[2]
        dx = [dx1, dx2, dx3]
        return dx


class RosslerAttractor(ODE_Autonomous):
    """
    Rössler attractor
    https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
    """

    def parameters(self):
        super().parameters()
        self.a = 0.2
        self.b = 0.2
        self.c = 5.7
        self.x0 = [0, 0, 0]
        self.nx = 3

    def equations(self, x, t):
        # Derivatives
        dx1 = - x[1] - x[2]
        dx2 = x[0] + self.a*x[1]
        dx3 = self.b + x[2]*(x[0]-self.c)
        dx = [dx1, dx2, dx3]
        return dx


class LotkaVolterra(ODE_Autonomous):
    """
    Lotka–Volterra equations, also known as the predator–prey equations
    https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
    """

    def parameters(self):
        self.a = 1.
        self.b = 0.1
        self.c = 1.5
        self.d = 0.75
        self.x0 = [5, 100]
        self.nx = 2

    def equations(self, x, t):
        # Derivatives
        dx1 = self.a*x[0] - self.b*x[0]*x[1]
        dx2 = -self.c*x[1] + self.d*self.b*x[0]*x[1]
        dx = [dx1, dx2]
        return dx


class Brusselator1D(ODE_Autonomous):
    """
    Brusselator
    https://en.wikipedia.org/wiki/Brusselator
    """

    def parameters(self):
        self.a = 1.0
        self.b = 3.0
        self.x0 = [1.0, 1.0]
        self.nx = 2

    def equations(self, x, t):
        # Derivatives
        dx1 = self.a + x[1]*x[0]**2 -self.b*x[0] - x[0]
        dx2 = self.b*x[0] - x[1]*x[0]**2
        dx = [dx1, dx2]
        return dx


class ChuaCircuit(ODE_Autonomous):
    """
    Chua's circuit
    https://en.wikipedia.org/wiki/Chua%27s_circuit
    https://www.chuacircuits.com/matlabsim.php
    """

    def parameters(self):
        super().parameters()
        self.a = 15.6
        self.b = 28.0
        # self.R = 1.0
        # self.C = 1.0
        self.m0 = -1.143
        self.m1 = -0.714
        self.x0 = [0.7, 0.0, 0.0]
        self.nx = 3

    def equations(self, x, t):
        fx = self.m1*x[0] + 0.5*(self.m0 - self.m1)*(np.abs(x[0] + 1) - np.abs(x[0] - 1))
        # Derivatives
        dx1 = self.a*(x[1] - x[0] - fx)
        dx2 = x[0] - x[1] + x[2]
        dx3 = -self.b*x[1]
        dx = [dx1, dx2, dx3]
        return dx


class Duffing(ODE_Autonomous):
    """
    Duffing equation
    https://en.wikipedia.org/wiki/Duffing_equation
    """

    def parameters(self):
        super().parameters()
        self.delta = 0.02
        self.alpha = 1
        self.beta = 5
        self.gamma = 8
        self.omega = 0.5
        self.x0 = [1.0, 0.0]
        self.nx = 2

    def equations(self, x, t):
        # Derivatives
        dx1 = x[1]
        dx2 = - self.delta*x[1] - self.alpha*x[0] - self.beta*x[0]**3 + self.gamma*np.cos(self.omega*t)
        dx = [dx1, dx2]
        return dx



