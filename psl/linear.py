"""
Autonomous and Non-autonomous linear ODEs and SSMs
"""
import numpy as np
import control

#  local imports
from psl.emulator import EmulatorBase


class ExpGrowth(EmulatorBase):
    """
    exponentia growth linear ODE
    https://en.wikipedia.org/wiki/Exponential_growth
    """

    def parameters(self):
        self.x0 = 1
        self.nx = 1
        self.k = 2
        self.A = self.k*np.eye(self.nx)

    def equations(self, x):
        x = self.A*x
        return x

    def simulate(self, ninit=0, nsim=1000, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param x: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response trajectories,  X
        """
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0
        X= []
        for k in range(nsim):
            x = self.equations(x)
            X.append(x)  # updated states trajectories
        return {'X': np.asarray(X)}


"""
Hybrid linear ODEs
CartPole, bauncing ball
"""


class LinCartPole(EmulatorBase):
    """
    Linearized Hybrid model of Inverted pendulum
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
    """

    def parameters(self):
        self.M = 0.5
        self.m = 0.2
        self.b = 0.1
        self.I = 0.006
        self.g = 9.8
        self.l = 0.3
        self.p = self.I*(self.M+self.m)+self.M*self.m*self.l**2; # denominator for the A and B matrices
        self.theta1 = -(self.I+self.m*self.l**2)*self.b/self.p
        self.theta2 = (self.m**2*self.g*self.l**2)/self.p
        self.theta3 = -(self.m*self.l*self.b)/self.p
        self.theta4 = (self.m*self.g*self.l*(self.M+self.m)/self.p)
        self.A = np.asarray([[0,1,0,0],[0,self.theta1,self.theta2,0],
                             [0,0,0,1],[0,self.theta3,self.theta4,0]])
        self.B = np.asarray([[0],[(self.I+self.m*self.l**2)/self.p],
                            [0],[self.m*self.l/self.p]])
        self.C = np.asarray([[1,0,0,0],[0,0,1,0]])
        self.D = np.asarray([[0],[0]])
        self.ssm = control.StateSpace(self.A, self.B, self.C, self.D)
        self.ssmd = self.ssm.sample(self.ts, method='euler')

        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nu = self.B.shape[1]
        self.x0 = np.asarray([0,0,-1,0])

    def equations(self, x, u):
        # Inputs (1): u is the force applied to the cart
        # States (4):
        # x1 position of the cart,
        # x2 velocity of the cart,
        # x3 angle of the pendulum relative to the cart
        # x4 rate of angle change
        x = np.matmul(np.asarray(self.ssmd.A), x) + np.matmul(np.asarray(self.ssmd.B), u).T
        #  physical constraints: position between +-10
        if x[0] >= 10:
            x[0] = 10
            x[1] = 0
        if x[0] <= -10:
            x[0] = -10
            x[1] = 0
        # angle in between +- 2*pi radians = -+ 360 degrees
        x[3] = np.mod(x[3], 2*np.pi)
        # positive +180 degrees and negative direction -180 degrees
        if x[3] >= np.pi:
            x[3] = np.pi-x[3]
        if x[3] <= -np.pi:
            x[3] = -np.pi-x[3]

        y = np.matmul(np.asarray(self.ssmd.C), x)
        return x, y

    def simulate(self, ninit, nsim, U, ts=None, x0=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param x: (ndarray, shape=(self.nx)) Initial state. If not give will use internal state.
        :return: The response trajectories,  X
        """
        if x0 is None:
            x = self.x0
        else:
            assert x0.shape[0] == self.nx, "Mismatch in x0 size"
            x = x0

        X, Y = [], []
        N = 0
        for u in U:
            x, y = self.equations(x, u)
            X.append(x)  # updated states trajectories
            Y.append(y)  # updated states trajectories
            N += 1
            if N == nsim:
                break
        Xout = np.asarray(X)
        Yout = np.asarray(Y)
        return {'X': Xout, 'Y': Yout}
