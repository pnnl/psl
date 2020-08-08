"""
Building thermal envelope models

# TODO: check loaded Ts value - not correct for some building models

"""
import os

import numpy as np
from scipy.io import loadmat

# local imports
from psl.emulator import SSM
from psl.perturb import Periodic, RandomWalk


class BuildingEnvelope(SSM):
    """
    building envelope heat transfer model
    linear building envelope dynamics and bilinear heat flow input dynamics
    different building types are stored in ./emulators/buildings/*.mat
    models obtained from: https://github.com/drgona/BeSim
    """
    def __init__(self, nsim=1000, ninit=1000, system='Reno_full', linear=True):
        self.system = system
        self.linear = linear
        self.resource_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters/buildings')
        super().__init__(nsim=nsim, ninit=ninit)

    # parameters of the dynamical system
    def parameters(self, system='Reno_full', linear=True):
        # file paths for different building models
        systems = {'SimpleSingleZone': os.path.join(self.resource_path, 'SimpleSingleZone.mat'),
                   'Reno_full': os.path.join(self.resource_path, 'Reno_full.mat'),
                   'Reno_ROM40': os.path.join(self.resource_path, 'Reno_ROM40.mat'),
                   'RenoLight_full': os.path.join(self.resource_path, 'RenoLight_full.mat'),
                   'RenoLight_ROM40': os.path.join(self.resource_path, 'RenoLight_ROM40.mat'),
                   'Old_full': os.path.join(self.resource_path, 'Old_full.mat'),
                   'Old_ROM40': os.path.join(self.resource_path, 'Old_ROM40.mat'),
                   'HollandschHuys_full': os.path.join(self.resource_path, 'HollandschHuys_full.mat'),
                   'HollandschHuys_ROM100': os.path.join(self.resource_path, 'HollandschHuys_ROM100.mat'),
                   'Infrax_full': os.path.join(self.resource_path, 'Infrax_full.mat'),
                   'Infrax_ROM100': os.path.join(self.resource_path, 'Infrax_ROM100.mat')
                   }
        self.system = system
        self.linear = linear  # if True use only linear building envelope model with Q as U
        file_path = systems[self.system]
        file = loadmat(file_path)

        #  LTI SSM model
        self.A = file['Ad']
        self.B = file['Bd']
        self.C = file['Cd']
        self.E = file['Ed']
        self.G = file['Gd']
        self.F = file['Fd']
        #  constraints bounds
        self.ts = file['Ts']  # sampling time
        self.umax = file['umax'].squeeze()  # max heat per zone
        self.umin = file['umin'].squeeze() # min heat per zone
        if not self.linear:
            self.dT_max = file['dT_max']  # maximal temperature difference deg C
            self.dT_min = file['dT_min']  # minimal temperature difference deg C
            self.mf_max = file['mf_max'].squeeze()  # maximal nominal mass flow l/h
            self.mf_min = file['mf_min'].squeeze()  # minimal nominal mass flow l/h
            #   heat flow equation constants
            self.rho = 0.997  # density  of water kg/1l
            self.cp = 4185.5  # specific heat capacity of water J/(kg/K)
            self.time_reg = 1 / 3600  # time regularization of the mass flow 1 hour = 3600 seconds
            # building type
        self.type = file['type']
        self.HC_system = file['HC_system']
        # problem dimensions
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.nq = self.B.shape[1]
        self.nd = self.E.shape[1]
        if self.linear:
            self.nu = self.nq
        else:
            self.n_mf = self.B.shape[1]
            self.n_dT = self.dT_max.shape[0]
            self.nu = self.n_mf + self.n_dT
        # initial conditions and disturbance profiles
        if self.system == 'SimpleSingleZone':
            self.x0 = file['x0'].reshape(self.nx)
        else:
            self.x0 = 0*np.ones(self.nx, dtype=np.float32)  # initial conditions
        self.D = file['disturb'] # pre-defined disturbance profiles
        #  steady states - linearization offsets
        self.x_ss = file['x_ss']
        self.y_ss = file['y_ss']
        # default simulation setup
        self.ninit = 0
        self.nsim = np.min([8640, self.D.shape[0]])
        if self.linear:
            self.U = Periodic(nx=self.nu, nsim=self.nsim, numPeriods=21, xmax=self.umax/2, xmin=self.umin, form='sin')
        else:
            self.M_flow = self.mf_max/2+RandomWalk(nx=self.n_mf, nsim=self.nsim, xmax=self.mf_max/2, xmin=self.mf_min, sigma=0.05)
            # self.M_flow = Periodic(nx=self.n_mf, nsim=self.nsim, numPeriods=21, xmax=self.mf_max, xmin=self.mf_min, form='sin')
            # self.DT = Periodic(nx=self.n_dT, nsim=self.nsim, numPeriods=15, xmax=self.dT_max/2, xmin=self.dT_min, form='cos')
            self.DT = RandomWalk(nx=self.n_dT, nsim=self.nsim, xmax=self.dT_max*0.6, xmin=self.dT_min, sigma=0.05)
            self.U = np.hstack([self.M_flow, self.DT])

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

