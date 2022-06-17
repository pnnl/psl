"""
Non-autonomous dynamic systems.

Chaotic nonlinear ODEs
    + https://en.wikipedia.org/wiki/List_of_chaotic_maps
"""
import numpy as np
import inspect, sys
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import odeint

from psl.emulator import EmulatorBase
from psl.perturb import Steps, Step, SplineSignal, Periodic, RandomWalk


class ODE_NonAutonomous(EmulatorBase):
    """
    base class autonomous ODE
    """

    def simulate(self, U=None, ninit=None, nsim=None, ts=None, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param ninit: (float) initial simulation time
        :param ts: (float) step size, sampling time
        :param x0: (float) state initial conditions
        :return: X, Y, U, D
        """
        if ninit is None:
            ninit = self.ninit
        if nsim is None:
            nsim = self.nsim
        if ts is None:
            ts = self.ts
        if U is None:
            U = self.U

        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        t = np.arange(0, nsim+1) * ts + ninit
        X = [x]
        N = 0
        for u in U:
            dT = [t[N], t[N + 1]]
            xdot = odeint(self.equations, x, dT, args=(u,))
            x = xdot[-1]
            X.append(x)
            N += 1
            if N == nsim-1:
                break
        Yout = np.asarray(X).reshape(nsim, -1)
        Uout = np.asarray(U).reshape(nsim, -1)
        return {'Y': Yout, 'U': Uout, 'X': np.asarray(X)}


class LorenzControl(ODE_NonAutonomous):

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.sigma = 10.
        self.beta = 2.66667
        self.rho = 28
        self.x0 = np.asarray([-8., 8., 27.])
        self.nx = 3
        self.nu = 2
        self.ts = 0.01
        self.ninit = 0.
        self.nsim = 1000
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        t = np.random.uniform(low=0, high=np.pi)
        self.ninit = t
        T = np.arange(t, t + self.ts * nsim, self.ts)
        return self.u_fun(T).T[:nsim]

    def equations(self, x, t, u):
        return [
            self.sigma * (x[1] - x[0]) + u[0],
            x[0] * (self.rho - x[2]) - x[1],
            x[0] * x[1] - self.beta * x[2] - u[1],
        ]

    def u_fun(self, t):
        return np.array([np.sin(2 * t), np.sin(8 * t)])


class SEIR_population(ODE_NonAutonomous):
    """
    Susceptible, Exposed, Infected, and Recovered (SEIR) population population model. Used to model COVID-19 spread.
    Source of the model:

    + https://apmonitor.com/do/index.php/Main/COVID-19Response

    states:

    + Susceptible (s): population fraction that is susceptible to the virus
    + Exposed (e): population fraction is infected with the virus but does not transmit to others
    + Infectious (i): population fraction that is infected and can infect others
    + Recovered (r): population fraction recovered from infection and is immune from further infection
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.N = 10000 # population
        # initial number of infected and recovered individuals
        self.e_0 = 1 / self.N
        self.i_0 = 0.00
        self.r_0 = 0.00
        self.s_0 = 1 - self.e_0 - self.i_0 - self.r_0
        self.x0 = np.asarray([self.s_0, self.e_0, self.i_0, self.r_0])

        self.nx = 4
        self.nu = 1

        self.t_incubation = 5.1
        self.t_infective = 3.3
        self.R0 = 2.4
        self.alpha = 1 / self.t_incubation
        self.gamma = 1 / self.t_infective
        self.beta = self.R0 * self.gamma
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        return Steps(nx=self.nu, nsim=nsim, values=None,
                     randsteps=int(np.ceil(self.nsim / 100)), xmax=1, xmin=0)

    def equations(self, x, t, u):
        """

        +  Inputs (1): social distancing (u=0 (none), u=1 (total isolation))
        +  States (4):
        +  Susceptible (s): population fraction that is susceptible to the virus
        +  Exposed (e): population fraction is infected with the virus but does not transmit to others
        +  Infectious (i): population fraction that is infected and can infect others
        +  Recovered (r): population fraction recovered from infection and is immune from further infection
        """

        s = x[0]
        e = x[1]
        i = x[2]
        u = u[0]

        sdt = -(1 - u) * self.beta * s * i
        edt = (1 - u) * self.beta * s * i - self.alpha * e
        idt = self.alpha * e - self.gamma * i
        rdt = self.gamma * i
        dx = [sdt, edt, idt, rdt]
        return dx


class Tank(ODE_NonAutonomous):
    """
    Single Tank model
    Original code obtained from APMonitor:

    + https://apmonitor.com/pdc/index.php/Main/TankLevel
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.rho = 1000.0  # water density (kg/m^3)
        self.A = 1.0  # tank area (m^2)
        self.x0 = 0
        self.U = self.get_U(self.nsim)
        self.nu = 2
        self.nx = 1

    def get_U(self, nsim):
        c = Steps(nx=1, nsim=self.nsim, values=None,
                  randsteps=int(np.ceil(nsim / 400)), xmax=55, xmin=45)
        valve = Steps(nx=1, nsim=nsim, values=None,
                      randsteps=int(np.ceil(nsim / 400)), xmax=100, xmin=0)
        return np.vstack([c.T, valve.T]).T

    def equations(self, x, t, u):
        """

        + States (1): level in the tanks
        + Inputs u(1): c - valve coefficient (kg/s / %open)
        + Inputs u(2): valve in % [0-100]
        """

        c = u[0]
        valve = u[1]
        dx_dt = (c / (self.rho*self.A)) * valve
        if x >= 1.0 and dx_dt > 0.0:
            dx_dt = 0
        return dx_dt


class TwoTank(ODE_NonAutonomous):
    """
    Two Tank model
    Original code obtained from APMonitor:

    + https://apmonitor.com/do/index.php/Main/LevelControl
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.ts = 1
        self.c1 = 0.08  # inlet valve coefficient
        self.c2 = 0.04  # tank outlet coefficient

        self.x0 = np.asarray([0, 0])
        self.U = self.get_U(self.nsim)
        self.nu = 2
        self.nx = 2

    def get_U(self, nsim):
        pump = Steps(nx=1, nsim=nsim, values=None,
                     randsteps=int(np.ceil(nsim / 200)), xmax=0.5, xmin=0)
        valve = Steps(nx=1, nsim=nsim, values=None,
                      randsteps=int(np.ceil(nsim / 300)), xmax=0.4, xmin=0)
        return np.vstack([pump.T, valve.T]).T

    def equations(self, x, t, u):
        # States (2): level in the tanks
        h1 = x[0]
        h2 = x[1]
        # Inputs (2): pump and valve
        pump = u[0]
        valve = u[1]
        dhdt1 = self.c1 * (1.0 - valve) * pump - self.c2 * np.sqrt(h1)
        dhdt2 = self.c1 * valve * pump + self.c2 * np.sqrt(h1) - self.c2 * np.sqrt(h2)
        if h1 >= 1.0 and dhdt1 > 0.0:
            dhdt1 = 0
        if h2 >= 1.0 and dhdt2 > 0.0:
            dhdt2 = 0
        dhdt = [dhdt1, dhdt2]
        return dhdt


class CSTR(ODE_NonAutonomous):
    """
    CSTR model
    Original code obtained from APMonitor:

    + http://apmonitor.com/do/index.php/Main/NonlinearControl
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        # Volumetric Flowrate (m^3/sec)
        self.q = 100
        # Volume of CSTR (m^3)
        self.V = 100
        # Density of A-B Mixture (kg/m^3)
        self.rho = 1000
        # Heat capacity of A-B Mixture (J/kg-K)
        self.Cp = 0.239
        # Heat of reaction for A->B (J/mol)
        self.mdelH = 5e4
        # E - Activation energy in the Arrhenius Equation (J/mol)
        # R - Universal Gas Constant = 8.31451 J/mol-K
        self.EoverR = 8750
        # Pre-exponential factor (1/sec)
        self.k0 = 7.2e10
        # U - Overall Heat Transfer Coefficient (W/m^2-K)
        # A - Area - this value is specific for the U calculation (m^2)
        self.UA = 5e4
        # Steady State Initial Conditions for the States
        self.Ca_ss = 0.87725294608097
        self.T_ss = 324.475443431599
        self.x0 = np.empty(2)
        self.x0[0] = self.Ca_ss
        self.x0[1] = self.T_ss
        # Steady State Initial Condition for the Uncontrolled Inputs
        self.u_ss = 300.0  # cooling jacket Temperature (K)
        self.Tf = 350  # Feed Temperature (K)
        self.Caf = 1  # Feed Concentration (mol/m^3)
        self.nx = 2
        self.nu = 1
        self.nd = 2
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        return self.u_ss + Steps(nx=1, nsim=nsim, values=None,
                                 randsteps=int(np.ceil(nsim / 100)), xmax=6, xmin=-6)

    def equations(self, x, t, u):
        # Inputs (1):
        # Temperature of cooling jacket (K)
        Tc = u
        # Disturbances (2):
        # Tf = Feed Temperature (K)
        # Caf = Feed Concentration (mol/m^3)
        # States (2):
        # Concentration of A in CSTR (mol/m^3)
        Ca = x[0]
        # Temperature in CSTR (K)
        T = x[1]

        # reaction rate
        rA = self.k0 * np.exp(-self.EoverR / T) * Ca
        # Calculate concentration derivative
        dCadt = self.q / self.V * (self.Caf - Ca) - rA
        # Calculate temperature derivative
        dTdt = self.q / self.V * (self.Tf - T) \
               + self.mdelH / (self.rho * self.Cp) * rA \
               + self.UA / self.V / self.rho / self.Cp * (Tc - T)
        xdot = np.zeros(2)
        xdot[0] = dCadt
        xdot[1] = dTdt
        return xdot


class InvPendulum(ODE_NonAutonomous):
    """
    Inverted Pendulum dynamics
    states: x = [\theta \dot{\theta}]; \theta is angle from upright equilibrium
    input: u = input torque

    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 2  # Number of states
        self.nu = 1  # Number of control inputs
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.L = 0.5  # length of the pole in m
        self.m = 0.15  # ball mass in kg
        self.b = 0.1  # friction
        self.x0 = [0.5, 0.0]

        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        return np.zeros(nsim)

    def equations(self, x, t, u):
        y = [x[1],
            (self.m * self.g * self.L * np.sin(x[0]) - self.b * x[1]) / (self.m * self.L ** 2)]
        y[1] = y[1] + (u / (self.m * self.L ** 2))
        return y


class UAV3D_kin(ODE_NonAutonomous):
    """
    Dubins 3D model -- UAV kinematic model with no wind
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 4    # Number of states
        self.nu = 3    # Number of control inputs
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.vmin = 9.5   # Minimum airspeed for stable flight (m/s)
        self.h = 10    # Minimum altitude to avoid crash (m)
        self.ts = 0.1

        # Initial Conditions for the States
        self.x0 = np.array([5, 10, 50, 0])
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        seed = 1
        headVec = np.multiply([0.0, -120.0, 0.0, 45.0, -90.0, 90.0, -175.0, 25.0, -90.0, 40.0, -20.0],
                              np.pi / 180.0)
        gammVec = np.multiply([0.0, 5.0, 10.0, -5.0, 0.0, 9.0, -1.0, 0.0, 3.0, -1.0, 0.0, 3.0, -1.0], np.pi / 180.0)
        velVec = [9.5, 16.0, 10.0, 11.0, 10.0, 11.5, 9.5, 15.0, 16.0, 13.0, 12.0, 12.0, 9.0]
        self.V = SplineSignal(nsim=nsim, values=velVec, xmin=9, xmax=15, rseed=seed)
        heading = SplineSignal(nsim=nsim, values=headVec, xmin=-20 * np.pi / 180, xmax=20 * np.pi / 180,
                               rseed=seed)
        self.phi = np.append([0.0], np.diff(heading) / self.ts)
        self.gamma = SplineSignal(nsim=nsim, values=gammVec, xmin=-10 * np.pi / 180, xmax=10 * np.pi / 180,
                                  rseed=seed)

        U1 = np.multiply(self.V, np.cos(self.gamma))
        U2 = np.multiply(self.V, np.sin(self.gamma))
        U3 = self.g * np.divide(np.tan(self.phi), self.V)
        return np.vstack([U1, U2, U3]).T

    def equations(self, x, t, u):
        """
        + States (4): [x, y, z]
        + Inputs (3): [V, phi, gamma]
        + Transformed Inputs (3): [U1, U2, U3]
        """

        U1 = u[0]
        U2 = u[1]
        U3 = u[2]

        dx_dt = np.zeros(4)
        dx_dt[0] = U1 * np.cos(x[3])
        dx_dt[1] = U1 * np.sin(x[3])
        dx_dt[2] = U2
        if x[2] <= self.h:
            dx_dt[2] = 0.0
        dx_dt[3] = U3

        return dx_dt


class UAV2D_kin(ODE_NonAutonomous):
    """
    Dubins 2D model -- UAV kinematic model with no wind
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 4    # Number of states
        self.nu = 1    # Number of control inputs
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.vmin = 9.5   # Minimum airspeed for stable flight (m/s)
        self.h = 10    # Minimum altitude to avoid crash (m)
        self.ts = 0.1

        self.V = 10   # Constant velocity  (m/s^2)

        self.x0 = np.array([5, 10, 10, 0])

        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):

        seed = 3
        self.phi = SplineSignal(nsim, values=None, xmin=-45*np.pi/180, xmax=45*np.pi/180, rseed=seed)
        return self.phi

    def equations(self, x, t, u):
        """
        + States (3): [x, y, z, psi]
        + Inputs (1): [phi]
        """

        V = self.V
        phi = u

        dx_dt = np.zeros(4)
        dx_dt[0] = V * np.cos(x[3])
        dx_dt[1] = V * np.sin(x[3])
        dx_dt[2] = 0.0
        dx_dt[3] = (self.g/V) * (np.tan(phi))

        return dx_dt


class UAV3D_reduced(ODE_NonAutonomous):
    """
    Reduced Dubins 3D model -- UAV kinematic model with transformed inputs
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 3    # Number of states
        self.nu = 3    # Number of control inputs
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.vmin = 9.5   # Minimum airspeed for stable flight (m/s)
        self.h = 10    # Minimum altitude to avoid crash (m)
        self.ts = 0.1

        self.x0 = np.array([5, 10, 50])

        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        seed = 2
        headVec = np.multiply([0.0, -120.0, 0.0, 45.0, -90.0, 90.0, -175.0, 25.0, -90.0, 40.0, -20.0], np.pi/180.0)
        gammVec = np.multiply([0.0, 5.0, 10.0, -5.0, 0.0, 9.0, -1.0, 0.0, 3.0, -1.0, 0.0], np.pi/180.0)
        self.V = SplineSignal(nsim=nsim, values=None, xmin=9, xmax=15, rseed=seed)
        self.phi = SplineSignal(nsim=nsim, values=None, xmin=-20*np.pi/180, xmax=20*np.pi/180, rseed=seed)
        self.gamma = SplineSignal(nsim=nsim, values=gammVec, xmin=-10*np.pi/180, xmax=10*np.pi/180, rseed=seed)

        U1 = np.multiply(self.V, np.cos(self.gamma))
        U2 = np.multiply(self.V, np.sin(self.gamma))
        U3 = SplineSignal(nsim=nsim, values=headVec * 3, xmin=-20 * np.pi / 180, xmax=20 * np.pi / 180, rseed=seed)

        return np.vstack([U1, U2, U3]).T

    def equations(self, x, t, u):
        """
        + States (4): [x, y, z]
        + Inputs (3): [V, phi, gamma]
        + Transformed Inputs (3): [U1, U2, U3]
        """

        U1 = u[0]
        U2 = u[1]
        U3 = u[2]

        dx_dt = np.zeros(3)
        dx_dt[0] = U1 * np.cos(U3)
        dx_dt[1] = U1 * np.sin(U3)
        dx_dt[2] = U2
        if x[2] <= self.h:
            dx_dt[2] = 0.0

        return dx_dt

      
class UAV3D_dyn(ODE_NonAutonomous):
    """
    UAV dynamic guidance model with no wind
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 6    # Number of states
        self.nu = 3    # Number of control inputs
        self.g = 9.81  # Acceleration due to gravity (m/s^2)
        self.W = 10.0  # Weight of the aircraft (kg)
        self.rho = 1.2 # Air density at sea level (kg/m^3); varies with altitude
        self.S = 10.0  # Wing area (m^2)
        self.lenF = 2  # Fuselage length
        self.b = 7.0  # Wingspan (m)
        self.AR = self.b**2 / self.S  # Aspect Ratio of the wing
        self.eps = 0.92  # Oswald efficiency factor (from Atlantik Solar UAV)
        self.K = 1 / (self.eps * np.pi * self.AR)  # aerodynamic coefficient
        self.ts = 0.1

        self.x0 = np.array([5, 10, 15, 0, np.pi/18, 9])
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        seed = 2
        loadVec = [1.0, 1.5, 1.0, 0.5, 0.1, 1.5, 1.0, 1.25, 1.0, 0.9, 1.0]
        T = SplineSignal(nsim=nsim, values=None, xmin=100, xmax=500, rseed=seed)
        phi = SplineSignal(nsim=nsim, values=None, xmin=-10 * np.pi / 180, xmax=10 * np.pi / 180, rseed=seed)
        load = SplineSignal(nsim=nsim, values=loadVec, xmin=0, xmax=3, rseed=seed)
        return np.vstack([T, phi, load]).T

    def equations(self, x, t, U):
        """
        + States (6): [x, y, z, psi, gamma, V]
        + Inputs (3): [T, phi, load]
        + load = Lift force / Weight

        """

        V = x[5]
        T = U[0]
        phi = U[1]
        load = 1.0
        CD0 = 0.015  # 0.074 * Re ** (-0.2)  # parasitic drag
        CL = 2 * load * self.W / self.rho * V ** 2 * self.S  # Lift coefficient
        CD = CD0 + self.K * CL**2     # Drag coefficient
        drag = self.rho * V ** 2 * self.S * CD   # Total drag

        dx_dt = np.zeros(6)

        dx_dt[0] = V * np.cos(x[3]) * np.cos(x[4])
        dx_dt[1] = V * np.sin(x[3]) * np.cos(x[4])
        dx_dt[2] = V * np.sin(x[4])
        dx_dt[3] = (self.g/V) * (load * np.sin(phi)) / np.cos(x[4])
        dx_dt[4] = (self.g/V) * (load * np.cos(phi) - np.cos(x[4]))
        dx_dt[5] = self.g * ((T - drag)/self.W - np.sin(x[4]))

        return dx_dt

      
"""
Chaotic nonlinear ODEs 

https://en.wikipedia.org/wiki/List_of_chaotic_maps
"""


class HindmarshRose(ODE_NonAutonomous):
    """
    Hindmarsh–Rose model of neuronal activity

    + https://en.wikipedia.org/wiki/Hindmarsh%E2%80%93Rose_model
    + https://demonstrations.wolfram.com/HindmarshRoseNeuronModel/
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.a = 1
        self.b = 2.6
        self.c = 1
        self.d = 5
        self.s = 4
        self.xR = -8/5
        self.r = 0.01
        self.umin = -10
        self.umax = 10
        self.x0 = np.asarray([-5,-10,0])
        self.U = self.get_U(self.nsim)
        self.nu = 1
        self.nx = 3

    def get_U(self, nsim):
        return 3 * np.asarray([np.ones((nsim))]).T


    def equations(self, x, t, u):
        theta = -self.a*x[0]**3 + self.b*x[0]**2
        phi = self.c -self.d*x[0]**2
        dx1 = x[1] + theta - x[2] + u
        dx2 = phi - x[1]
        dx3 = self.r*(self.s*(x[0]-self.xR)-x[2])
        dx = [dx1, dx2, dx3]
        return dx


class Iver_kin_reduced(ODE_NonAutonomous):
    """
    Kinetic model of Unmanned Underwater Vehicle (Yan et al 2020) -- UAV kinematic model with **no roll**
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=3):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 5
        self.nu = 5
        self.ts = 0.1
        self.K_alpha = 100.0 # Barrier function parameter

        self.x0 = np.array([0, 0, 0, 0, 0])
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        seed = self.seed
        u = Steps(nsim=nsim, values=None, randsteps=20, xmin=1.0, xmax=3.0, rseed=seed).T
        v = Steps(nsim=nsim, values=None, randsteps=20, xmin=-0.5, xmax=0.5, rseed=2 * seed).T
        w = Steps(nsim=nsim, values=None, randsteps=20, xmin=-0.5, xmax=0.5, rseed=3 * seed).T
        q = Steps(nsim=nsim, values=None, randsteps=20, xmin=-0.5, xmax=0.5, rseed=4 * seed).T
        r = Steps(nsim=nsim, values=None, randsteps=20, xmin=-0.5, xmax=0.5, rseed=5 * seed).T

        return np.vstack( [u, v, w, q, r] ).T


    def equations(self, x, t, u):
        """
        + States (5): [xi, eta, zeta, theta, psi]
        + Inputs (5): [u, v, w, q, r]
        """
        theta = x[3]
        psi = x[4]

        # Control
        uu = u[0]
        v = u[1]
        w = u[2]
        q = u[3]
        r = u[4]

        # Barrier function to avoid singularities
        h = (np.pi/2.0 - 0.01)**2 - theta**2
        z = -2.0*theta*q + self.K_alpha*h

        dx_dt = np.zeros(5)
        dx_dt[0] = np.cos( psi )*np.cos( theta )*uu - np.sin( psi )*v + np.sin( theta )*np.cos( psi )*w
        dx_dt[1] =  np.sin( psi )*np.cos( theta )*uu + np.cos( psi )*v + ( np.sin( theta )*np.sin( psi ) )*w
        dx_dt[2] = -np.sin( theta )*uu + np.cos( theta )*w
        dx_dt[3] = q - z
        dx_dt[4] = r/(np.cos( theta ))

        return dx_dt


class Iver_kin(ODE_NonAutonomous):
    """
    Kinetic model of Unmanned Underwater Vehicle (Fossen) -- Full UAV kinematic model
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=3):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 6
        self.nu = 6
        self.ts = 0.1

        self.x0 = np.array([1.0, 0, 0, 0, 0, 0])
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        seed = self.seed
        u = SplineSignal(nsim=nsim, values=None, xmin=1.0, xmax=3.0, rseed=seed)
        v = SplineSignal(nsim=nsim, values=None, xmin=-0.1, xmax=0.1, rseed=seed)
        w = SplineSignal(nsim=nsim, values=None, xmin=-0.1, xmax=0.1, rseed=seed)
        p = SplineSignal(nsim=nsim, values=None, xmin=-0.01, xmax=0.01, rseed=seed)
        q = SplineSignal(nsim=nsim, values=None, xmin=-0.01, xmax=0.01, rseed=seed)
        r = SplineSignal(nsim=nsim, values=None, xmin=-0.01, xmax=0.01, rseed=seed)

        return np.vstack( [u, v, w, p, q, r] ).T

    def equations(self, x, t, u):
        """
        + States (5): [n, e, d, phi, theta, psi]
        + Inputs (5): [u, v, w, p, q, r]
        """
        phi = x[3]
        theta = x[4]
        psi = x[5]

        # Control
        uu = u[0]
        v = u[1]
        w = u[2]
        p = u[3]
        q = u[4]
        r = u[5]

        dx_dt = np.zeros(6)
        dx_dt[0] = uu*np.cos( psi )*np.cos( theta ) + v*( np.cos( psi )*np.sin( theta )*np.sin( phi ) - np.sin( psi )*np.cos( phi ) ) + w*( np.sin( psi )*np.sin( phi ) + np.cos( psi )*np.cos( phi )*np.sin( theta ) )
        dx_dt[1] = uu*( np.sin( psi )*np.cos( theta ) ) + v*( np.cos( psi )*np.cos( phi ) + np.sin( phi )*np.sin( theta )*np.sin( psi ) ) + w*( np.sin( theta )*np.sin( psi )*np.cos( phi ) - np.cos( psi )*np.sin( phi ) )
        dx_dt[2] = -uu*np.sin( theta ) + v*np.cos( theta )*np.sin( phi ) + w*np.cos( theta )*np.cos( phi )
        dx_dt[3] = p + q*np.sin ( phi )*np.tan( theta ) + r*np.cos( phi )*np.tan( theta )
        dx_dt[4] = q*np.cos( phi ) - r*np.sin( phi )
        dx_dt[5] = q*( np.sin( phi )/np.cos( theta ) ) + r*( np.cos( phi )/np.cos( theta ) )

        return dx_dt

class Iver_dyn(ODE_NonAutonomous):
    """
    Dynamic model of Unmanned Underwater Vehicle (Fossen) -- Excludes hydrostatic/dynamic terms and ocean current
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=3):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 6    # Number of states
        self.nu = 6    # Number of control inputs
        self.ts = 0.1

        self.m = 1.0        # Mass of of the vehicle (kg)
        self.Ixx = 0.001    # Moment of inertia about resp. axes, written w.r.t body frame (kg m^2)
        self.Iyy = 0.001    # Moment of inertia about resp. axes, written w.r.t body frame (kg m^2)
        self.Izz = 0.001    # Moment of inertia about resp. axes, written w.r.t body frame (kg m^2)
        self.Ixy = 0.001    # Moment of inertia about resp. axes, written w.r.t body frame (kg m^2)
        self.Ixz = 0.001    # Moment of inertia about resp. axes, written w.r.t body frame (kg m^2)
        self.Iyz = 0.001    # Moment of inertia about resp. axes, written w.r.t body frame (kg m^2)
        self.xg = 0.0       # Center of mass w.r.t x axis, written in body frame (m)
        self.yg = 0.0       # Center of mass w.r.t y axis, written in body frame (m)
        self.zg = 0.0       # Center of mass w.r.t z axis, written in body frame (m)

        self.x0 = np.array([1.0, 0, 0, 0, 0, 0])
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        seed = self.seed
        tau_X = SplineSignal(nsim=nsim, values=None, xmin=1.0, xmax=3.0, rseed=seed)
        tau_Y = SplineSignal(nsim=nsim, values=None, xmin=-0.1, xmax=0.1, rseed=seed)
        tau_Z = SplineSignal(nsim=nsim, values=None, xmin=-0.1, xmax=0.1, rseed=seed)
        tau_K = SplineSignal(nsim=nsim, values=None, xmin=-0.01, xmax=0.01, rseed=seed)
        tau_M = SplineSignal(nsim=nsim, values=None, xmin=-0.01, xmax=0.01, rseed=seed)
        tau_N = SplineSignal(nsim=nsim, values=None, xmin=-0.01, xmax=0.01, rseed=seed)

        return np.vstack( [tau_X, tau_Y, tau_Z, tau_K, tau_M, tau_N] ).T

    def equations(self, x, t, u):
        """
        + States (6): [u, v, w, p, q, r]
        + Inputs (6): [tau_X, tau_Y, tau_Z, tau_K, tau_M, tau_N]
        """

        # States
        uu = x[0]
        v = x[1]
        w = x[2]
        p = x[3]
        q = x[4]
        r = x[5]
        nu = np.array(x)
        nu1 = np.array([uu, v, w])
        nu2 = np.array([p, q, r])

        rbg = np.array([self.xg, self.yg, self.zg ])
        Io = np.array([ [self.Ixx, -self.Ixy, -self.Ixz], [-self.Ixy, self.Iyy, -self.Iyz], [-self.Ixz, -self.Iyz, self.Izz] ])
        M_rb = np.block([ [self.m*np.eye(3), -self.m*self.Cross(rbg) ], [self.m*self.Cross(rbg), Io ] ])
        C_rb = np.block([ [np.zeros((3,3)), -self.m*self.Cross(nu1) - self.m*np.multiply( self.Cross(nu2), self.Cross(rbg)) ], [-self.m*self.Cross(nu1) + self.m*np.multiply( self.Cross(rbg), self.Cross(nu2) ), -self.Cross( Io.dot(nu2) )  ] ])

        dx_dt = np.linalg.inv(M_rb).dot( -C_rb.dot(nu) + u  )
        return dx_dt

    def Cross(self,x):
        """
        + Input: 3d array x
        + Output: cross product matrix (skew symmetric)
        """
        return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0] ])


class Iver_dyn_reduced(ODE_NonAutonomous):
    """
    Dynamic model of Unmanned Underwater Vehicle (Yan et al) -- Excludes rolling, includes hydrostate/dynamic terms, no currents
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=3):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 10    # Number of states
        self.nu = 5    # Number of control inputs
        self.ts = 0.1

        self.m = 100.0        # Mass of of the vehicle (kg)
        self.Iyy = 0.1    # Moment of inertia about resp. axes, written w.r.t body frame (kg m^2)
        self.Izz = 0.1    # Moment of inertia about resp. axes, written w.r.t body frame (kg m^2)
        self.Xu_dot = 0.0001  # Hydrodynamic coefficient
        self.Yv_dot = 0.0001  # Hydrodynamic coefficient
        self.Zw_dot = 0.0001  # Hydrodynamic coefficient
        self.Mq_dot = 0.0001  # Hydrodynamic coefficient
        self.Nr_dot = 0.0001  # Hydrodynamic coefficient
        self.Xu = 0.0001      # Hydrodynamic coefficient
        self.Yv = 0.0001      # Hydrodynamic coefficient
        self.Zw = 0.0001      # Hydrodynamic coefficient
        self.Mq = 0.0001      # Hydrodynamic coefficient
        self.Nr = 0.0001      # Hydrodynamic coefficient
        self.Xuu = 0.0001     # Hydrodynamic coefficient
        self.Yvv = 0.0001     # Hydrodynamic coefficient
        self.Zww = 0.0001     # Hydrodynamic coefficient
        self.Mqq = 0.0001     # Hydrodynamic coefficient
        self.Nrr = 0.0001     # Hydrodynamic coefficient
        self.V = 0.01       # Volume of water displaced by the vehicle
        self.rho = 1000    # Density of the water (kg/m^3)
        self.g = 9.81       # Acceleration due to gravity (m/s^2)
        self.GML = 0.01     # Vertical metacentric height (m)

        self.x0 = np.array([0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0])
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        seed = self.seed
        tau_X = SplineSignal(nsim=nsim, values=None, xmin=1.0, xmax=3.0, rseed=seed)
        tau_Y = SplineSignal(nsim=nsim, values=None, xmin=-0.1, xmax=0.1, rseed=seed)
        tau_Z = SplineSignal(nsim=nsim, values=None, xmin=-0.1, xmax=0.1, rseed=seed)
        tau_M = SplineSignal(nsim=nsim, values=None, xmin=-0.01, xmax=0.01, rseed=seed)
        tau_N = SplineSignal(nsim=nsim, values=None, xmin=-0.01, xmax=0.01, rseed=seed)

        return np.vstack( [tau_X, tau_Y, tau_Z, tau_M, tau_N] ).T

    def equations(self, x, t, u):
        """
        + States (10): [n, e, d, theta, psi, u, v, w, p, q, r]
        + Inputs (5): [tau_X, tau_Y, tau_Z, tau_M, tau_N]
        """

        n = x[0]
        e = x[1]
        d = x[2]
        theta = x[3]
        psi = x[4]
        uu = x[5]
        v = x[6]
        w = x[7]
        q = x[8]
        r = x[9]
        nu = np.array([ uu, v, w, q, r])

        # Kinematics: dot(eta) = J nu
        dx_dt = np.zeros(10)
        dx_dt[0] = np.cos(x[4]) * np.cos(x[3]) * nu[0] - np.sin(x[4]) * nu[1] + np.sin(x[3]) * np.cos(x[4]) * nu[2]
        dx_dt[1] = (np.sin(x[4]) * np.cos(x[3]) * np.cos(x[4]) * np.sin(x[3]) * np.sin(x[4]) - np.sin(x[3])) * nu[0]
        dx_dt[2] = np.cos(x[3]) * nu[2]
        dx_dt[3] = nu[3]
        dx_dt[4] = nu[4] / (np.cos(x[3]))

        # Construct dynamics in matrix form: dot(nu) = inv(M) ( -C nu - D nu - g + tau)
        M_rb = np.diag([ self.m, self.m, self.m, self.Iyy, self.Izz ])
        M_a = np.diag([ -self.Xu_dot, -self.Yv_dot, -self.Zw_dot, -self.Mq_dot, -self.Nr_dot ])
        M = M_rb + M_a
        C_rb = np.array([ [0, 0, 0, self.m*w, -self.m*v], [0, 0, 0, 0, self.m*uu], [0, 0, 0, -self.m*uu, 0], [-self.m*w, 0, self.m*uu, 0, 0], [self.m*v, -self.m*uu, 0, 0, 0] ])
        C_a = np.array([ [0, 0, 0, -self.Zw_dot*w, self.Yv_dot*v], [0, 0, 0, 0, -self.Xu_dot*uu], [0, 0, 0, self.Xu_dot*uu, 0],[self.Zw_dot*w, 0, -self.Xu_dot*uu, 0, 0], [-self.Yv_dot*v, self.Xu_dot*uu, 0, 0, 0] ])
        C = C_rb + C_a
        D = np.diag([ self.Xu, self.Yv, self.Zw, self.Mq, self.Nr ]) + np.diag([ self.Xuu, self.Yvv, self.Zww, self.Mqq, self.Nrr ])
        g = np.array([ 0, 0, 0, self.rho*self.V*self.GML*np.sin(theta), 0])

        dx_dt[5:] = np.linalg.inv(M).dot( -C.dot(nu) -D.dot(nu) - g + u  )

        return dx_dt


class Iver_dyn_simplified(ODE_NonAutonomous):
    """
    Dynamic model of Unmanned Underwater Vehicle (modified from Stankiewicz et al) -- Excludes rolling, sway, currents, Includes: hydrostate/dynamic terms, control surface deflections/propeller thrust, and actuator dynamics
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=3):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.nx = 12    # Number of states (including actuator dynamics)
        self.nu = 3    # Number of control inputs
        self.ts = 0.1

        self.Mq = -0.748        # Hydrodynamic coefficient (1/s)
        self.Nur = -0.441       # Hydrodynamic coefficient (1/m)
        self.Xuu = -0.179       # Hydrodynamic coefficient (1/m)
        self.Zww = 0.098        # Hydrodynamic coefficient (1/m)
        self.Muq = -3.519       # Hydrodynamic coefficient (1/m)
        self.WB = 0.0           # Out-of-ballast term based on weight and buoyancy ratio (m/s^2) (Differs from Stankiewicz, here set to zero for neutral buoyancy)
        self.Bz = 8.947         # Bouyancy term that accounts for the center of bouyancy vertical offset from the center of gravity (1/s^2)
        self.k = 0.519          # Hydrodynamic coefficient (m/s^2)
        self.b = 3.096          # Hydrodynamic coefficient (1/m^2)
        self.c = 0.065          # Hydrodynamic coefficient (1/m^2)
        self.K_delta_u = -10.0   # Thruster dynamic coefficient
        self.K_delta_q = -10.0   # Elevator deflection dynamic coefficient
        self.K_delta_r = -10.0   # Rudder deflection dynamic coefficient

        self.x0 = np.array([0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0])

        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        seed = self.seed
        delta_u = Steps(nsim=nsim, values=None, randsteps=100, xmin=0.0, xmax=1.0, rseed=seed).T
        delta_q = Steps(nsim=nsim, values=None, randsteps=100, xmin=-1.0, xmax=1.0, rseed=2 * seed).T
        delta_r = Steps(nsim=nsim, values=None, randsteps=100, xmin=-1.0, xmax=1.0, rseed=3 * seed).T
        return np.vstack( [delta_u, delta_q, delta_r] ).T

    def equations(self, x, t, u):
        """
        + States (12): [px, py, pz, theta, psi, uu, w, q, r, delta_u, delta_q, delta_r]
        + Inputs (3): [delta_uc, delta_qc, delta_rc] (thrust speed/deflections, normalized)
        """
        theta = x[3]
        psi = x[4]
        uu = x[5]
        w = x[6]
        q = x[7]
        r = x[8]
        delta_u = x[9]
        delta_q = x[10]
        delta_r = x[11]

        delta_uc = u[0]
        delta_qc = u[1]
        delta_rc = u[2]

        # Kinematics:
        dx_dt = np.zeros(12)
        dx_dt[0] = uu*np.cos(psi)*np.cos(theta) + w*np.cos(psi)*np.sin(theta)
        dx_dt[1] = uu*np.sin(psi)*np.cos(theta) + w*np.sin(psi)*np.sin(theta)
        dx_dt[2] = w*np.cos(theta) - uu*np.sin(theta)
        dx_dt[3] = q
        dx_dt[4] = r / (np.cos(theta))

        # Dynamics
        dx_dt[5] = self.Xuu*(uu**2) + self.k*delta_u
        dx_dt[6] = self.Zww*w*np.absolute(w) + self.WB*np.cos(theta)
        dx_dt[7] = self.Muq*uu*q + self.Mq*q - self.Bz*np.sin(theta) + self.b*(uu**2)*delta_q
        dx_dt[8] = self.Nur*uu*r + self.c*(uu**2)*delta_r

        # Actuator dynamics:
        dx_dt[9] = self.K_delta_u*( delta_u - delta_uc )
        dx_dt[10] = self.K_delta_q*( delta_q - delta_qc )
        dx_dt[11] = self.K_delta_r*( delta_r - delta_rc )

        return dx_dt

class Iver_dyn_simplified_output(ODE_NonAutonomous):
    """
    Dynamic model of Unmanned Underwater Vehicle (modified from Stankiewicz et al) -- Excludes rolling, sway, currents, Includes: hydrostate/dynamic terms, control surface deflections/propeller thrust, and actuator dynamics
    with non-kinematic output
    """

    # parameters of the dynamical system
    def parameters(self):
        self.nx = 9    # Number of states (including actuator dynamics)
        self.nu = 3    # Number of control inputs
        self.ts = 0.1

        # Model parameters
        self.Mq = -0.748        # Hydrodynamic coefficient (1/s)
        self.Nur = -0.441       # Hydrodynamic coefficient (1/m)
        self.Xuu = -0.179       # Hydrodynamic coefficient (1/m)
        self.Zww = 0.098        # Hydrodynamic coefficient (1/m)
        self.Muq = -3.519       # Hydrodynamic coefficient (1/m)
        self.WB = 0.0           # Out-of-ballast term based on weight and buoyancy ratio (m/s^2) (Differs from Stankiewicz, here set to zero for neutral buoyancy)
        self.Bz = 8.947         # Bouyancy term that accounts for the center of bouyancy vertical offset from the center of gravity (1/s^2)
        self.k = 0.519          # Hydrodynamic coefficient (m/s^2)
        self.b = 3.096          # Hydrodynamic coefficient (1/m^2)
        self.c = 0.065          # Hydrodynamic coefficient (1/m^2)
        self.K_delta_u = -10.0   # Thruster dynamic coefficient
        self.K_delta_q = -10.0   # Elevator deflection dynamic coefficient
        self.K_delta_r = -10.0   # Rudder deflection dynamic coefficient

        # Initial Conditions for the States
        self.x0 = np.array([0, 0, 0.01, 0, 0, 0, 0, 0, 0])

        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        seed = 3
        #delta_u = SplineSignal(nsim=self.nsim, values=None, xmin= 0.0, xmax=1.0, rseed=seed)
        #delta_q = SplineSignal(nsim=self.nsim, values=None, xmin=-1.0, xmax=1.0, rseed=2*seed)
        #delta_r = SplineSignal(nsim=self.nsim, values=None, xmin=-1.0, xmax=1.0, rseed=3*seed)

        delta_u = Steps(nsim=nsim, values=None, randsteps=100, xmin=0.0, xmax=1.0, rseed=seed).T
        delta_q = Steps(nsim=nsim, values=None, randsteps=100, xmin=-1.0, xmax=1.0, rseed=2 * seed).T
        delta_r = Steps(nsim=nsim, values=None, randsteps=100, xmin=-1.0, xmax=1.0, rseed=3 * seed).T

        return np.vstack( [delta_u, delta_q, delta_r] ).T


    # equations defining the dynamical system
    def equations(self, x, t, u):
        """
        + States (9): [theta, psi, uu, w, q, r, delta_u, delta_q, delta_r]
        + Inputs (3): [delta_uc, delta_qc, delta_rc] (thrust speed/deflections, normalized)
        """

        # States
        theta = x[0]
        psi = x[1]
        uu = x[2]
        w = x[3]
        q = x[4]
        r = x[5]
        delta_u = x[6]
        delta_q = x[7]
        delta_r = x[8]

        # Control
        delta_uc = u[0]
        delta_qc = u[1]
        delta_rc = u[2]

        # Kinematics:
        dx_dt = np.zeros(9)
        dx_dt[0] = q
        dx_dt[1] = r / (np.cos(theta))

        # Dynamics
        dx_dt[2] = self.Xuu*(uu**2) + self.k*delta_u
        dx_dt[3] = self.Zww*w*np.absolute(w) + self.WB*np.cos(theta)
        dx_dt[4] = self.Muq*uu*q + self.Mq*q - self.Bz*np.sin(theta) + self.b*(uu**2)*delta_q
        dx_dt[5] = self.Nur*uu*r + self.c*(uu**2)*delta_r

        # Actuator dynamics:
        dx_dt[6] = self.K_delta_u*( delta_u - delta_uc )
        dx_dt[7] = self.K_delta_q*( delta_q - delta_qc )
        dx_dt[8] = self.K_delta_r*( delta_r - delta_rc )

        return dx_dt


class SwingEquation(ODE_NonAutonomous):
    """
    Power Grid Swing Equation.

    + https://en.wikipedia.org/wiki/Swing_equation
    + The second-order swing equation is converted to
    + two first-order ODEs
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=3):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.Pm = 0.8  # Mechanical power
        self.Pmax = 5.0  # Maximum electrical output
        self.H = 500.0  # Inertia constant
        self.D = 5.0  # Damping coefficient
        self.freq = 60.0  # Base frequency
        self.ws = 2 * np.pi * self.freq  # Base angular speed
        self.M = 2 * self.H / self.ws  # scaled inertia constant
        self.nx = 2  # number of variables
        self.mode = 1  # 0 (Constant mechanical power and Pmax)
        # 1 (Noisy mechanical power with constant Pmax)
        # 2 (Constant mechanical power with fault on-off Pmax (square wave))

        if (self.mode == 0):
            self.delta0 = 0.0
        else:
            self.delta0 = np.arcsin(self.Pm / self.Pmax)  # initial condition for machine angle \delta

        self.dw0 = 0.0  # initial condition for machine speed deviation d\omega
        self.x0 = [self.delta0, self.dw0]  # initial condition for the model

        self.ts = 0.01  # Time-step
        self.ninit = 0.0  # Simulation start at t=0
        self.nsim = 1000  # 1000 steps = 10 sec. horizon
        self.tfaulton = 0.05
        self.tfaultoff = 0.15
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim):
        if (self.mode == 0):
            # 0 (Constant mechanical power and Pmax)
            Pmech = Step(nx=1, nsim=nsim, tstep=100, xmax=self.Pm, xmin=self.Pm, rseed=1)
            Pmax = Step(nx=1, nsim=nsim, tstep=100, xmax=self.Pmax, xmin=self.Pmax, rseed=1)
            self.U = np.vstack([Pmech.T, Pmax.T]).T
            self.nu = 2
        elif (self.mode == 1):
            # 1 (Noisy mechanical power with constant Pmax)
            Pmech = RandomWalk(nx=1, nsim=nsim, xmax=self.Pm * 1.02, xmin=self.Pm * 0.98, sigma=0.1, rseed=1)
            self.U = Pmech
            self.nu = 1
        else:
            # 2 (Constant mechanical power with fault on-off Pmax (square wave))
            Signal = []
            signal = []
            for tstep in range(0, nsim):
                t = tstep * self.ts
                if (t < self.tfaulton):  # Pre-fault
                    signal.append(self.Pmax)
                elif (t >= self.tfaulton and t <= self.tfaultoff):  # Fault-on
                    signal.append(0)
                else:  # Post-fault
                    signal.append(self.Pmax)
            Signal.append(signal)
            Pmax = np.asarray(Signal).T
            self.U = Pmax
            self.nu = 1
        return self.U

    def equations(self, x, t, u):
        delta = x[0]
        domega = x[1]

        if (self.mode == 0):
            Pm = u[0]
            Pmax = u[1]
        elif (self.mode == 1):
            Pm = u[0]
            Pmax = self.Pmax
        else:
            Pm = self.Pm
            Pmax = u[0]

        dx_dt = [self.ws * domega, (Pm - Pmax * np.sin(delta) - self.D * domega) / self.M]
        return dx_dt

systems = dict(inspect.getmembers(sys.modules[__name__], lambda x: inspect.isclass(x)))
systems = {k: v for k, v in systems.items() if issubclass(v, ODE_NonAutonomous) and v is not ODE_NonAutonomous}


