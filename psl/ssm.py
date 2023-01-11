import os

from scipy.io import loadmat
import numpy as np

from psl.emulator import EmulatorBase
from psl.perturb import Periodic, RandomWalk


class SSM(EmulatorBase):
    """
    base class state space model
    """

    def __init__(self, nsim=1001, ninit=0, ts=0.1, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, ts=ts, seed=seed)
        self.x_ss = 0.
        self.y_ss = 0.
        self.x0 = 0.

    def simulate(self, ninit=None, nsim=None, U=None, D=None, x0=None, Time=None):
        """
        :param nsim: (int) Number of steps for open loop response
        :param U: (ndarray, shape=(nsim, self.nu)) control signals
        :param D: (ndarray, shape=(nsim, self.nd)) measured disturbance signals
        :param x0: (ndarray, shape=(self.nx)) Initial state.
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
        X, Y = [x+self.x_ss], []
        for k in range(nsim):
            u = U[k, :] if U is not None else None
            d = D[k, :] if D is not None else None
            x, y = self.equations(x, u, d)
            X.append(x + self.x_ss)
            Y.append(y - self.y_ss)
        Xout = np.asarray(X).reshape(nsim+1, -1)
        Yout = np.asarray(Y).reshape(nsim, -1)
        Uout = np.asarray(U).reshape(nsim, self.nu) if U is not None else None
        if D is not None:
            if self.d_idx is not None:
                Dout = np.asarray(D[:, self.d_idx]).reshape(nsim, len(self.d_idx)) - self.d_ss
            else:
                Dout = np.asarray(D).reshape(nsim, self.nd)
        else:
            Dout = None
        return {'X': Xout, 'Y': Yout, 'U': Uout, 'D': Dout}


class BuildingEnvelope(SSM):
    """
    building envelope heat transfer model
    linear building envelope dynamics and bilinear heat flow input dynamics
    different building types are stored in ./emulators/buildings/*.mat
    documentation about the building systems is stored in ./emulators/buildings/building_type/*
    models obtained from: https://github.com/drgona/BeSim
    """
    systems = ['SimpleSingleZone', 'Reno_full', 'Reno_ROM40', 'RenoLight_full',
               'RenoLight_ROM40', 'Old_full', 'Old_ROM40',
               'HollandschHuys_full', 'HollandschHuys_ROM100', 'Infrax_full', 'Infrax_ROM100']

    T_dist_idx = {'Reno_full': [40], 'Reno_ROM40': [40],
                  'RenoLight_full': [40], 'RenoLight_ROM40': [40],
                  'Old_full': [40], 'Old_ROM40': [40],
                  'HollandschHuys_full': [221], 'HollandschHuys_ROM100':  [221],
                  'Infrax_full': [160], 'Infrax_ROM100': [160],
                  'SimpleSingleZone': None}

    def path(self, system):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters/buildings', f'{system}.mat')

    def __init__(self, nsim=1000, ninit=1000, system='Reno_full', linear=True, seed=59):
        super().__init__(nsim=nsim, ninit=ninit, seed=seed)
        self.seed = seed
        self.system = system
        self.linear = linear  # if True use only linear building envelope model with Q as U
        file = loadmat(self.path(system))
        #  LTI SSM model
        self.A = file['Ad']
        self.B = file['Bd']
        self.C = file['Cd']
        self.E = file['Ed']
        self.G = file['Gd']
        self.F = file['Fd']
        # problem dimensions
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nq = self.B.shape[1]
        self.nd = self.E.shape[1]
        # observable disturbance index
        self.d_idx = self.T_dist_idx[system]
        #  constraints bounds
        self.ts = file['Ts']  # sampling time
        self.umax = file['umax'].squeeze()   # max heat per zone
        self.umin = file['umin'].squeeze()   # min heat per zone
        if not self.linear:
            self.dT_max = file['dT_max']  # maximal temperature difference deg C
            self.dT_min = file['dT_min']  # minimal temperature difference deg C
            self.mf_max = file['mf_max'].reshape(self.nq,)  # maximal nominal mass flow l/h
            self.mf_min = file['mf_min'].reshape(self.nq,)  # minimal nominal mass flow l/h
            #   heat flow equation constants
            self.rho = 0.997  # density  of water kg/1l
            self.cp = 4185.5  # specific heat capacity of water J/(kg/K)
            self.time_reg = 1 / 3600  # time regularization of the mass flow 1 hour = 3600 seconds
            # building type
        self.type = file['type']
        self.HC_system = file['HC_system']


        if self.linear:
            self.nu = self.nq
        else:
            self.n_mf = self.B.shape[1]
            self.n_dT = self.dT_max.shape[0]
            self.nu = self.n_mf + self.n_dT

        if self.system == 'SimpleSingleZone':
            self.x0 = file['x0'].reshape(self.nx)
        else:
            self.x0 = 0*np.ones(self.nx, dtype=np.float32)  # initial conditions
        self.D = file['disturb']    # pre-defined disturbance profiles
        #  steady states - linearization offsets
        self.x_ss = file['x_ss']
        self.y_ss = file['y_ss']
        self.d_ss = np.asarray([273.15])
        self.ninit = 0
        self.nsim = np.min([nsim, self.D.shape[0]])
        self.U = self.get_U(self.nsim)

    def get_U(self, nsim, rseed=1):
        if self.linear:
            return Periodic(nx=self.nu, nsim=nsim, numPeriods=21, xmax=self.umax/2, xmin=self.umin, form='sin', rseed=rseed).astype(np.float32)
        else:
            self.M_flow = self.mf_max/2+RandomWalk(nx=self.n_mf, nsim=nsim, xmax=self.mf_max/2, xmin=self.mf_min, sigma=0.05, rseed=rseed)
            self.DT = RandomWalk(nx=self.n_dT, nsim=nsim, xmax=self.dT_max*0.6, xmin=self.dT_min, sigma=0.05, rseed=rsead)
            return np.hstack([self.M_flow, self.DT]).astype(np.float32)

    def get_x0(self, rand=False):
        if rand:
            return np.random.uniform(low=-20, high=5, size=self.nx).astype(np.float32)
        else:
            return np.zeros(self.nx, dtype=np.float32)

    def equations(self, x, u, d):
        if self.linear:
            q = u
        else:
            m_flow = u[0:self.n_mf]
            dT = u[self.n_mf:self.n_mf+self.n_dT]
            q = m_flow * self.rho * self.cp * self.time_reg * dT
        x = np.matmul(self.A, x) + np.matmul(self.B, q) + np.matmul(self.E, d) + self.G.ravel()
        y = np.matmul(self.C, x) + self.F.ravel()
        return x, y


systems = {k: BuildingEnvelope for k in BuildingEnvelope.systems}
