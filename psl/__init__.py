from .autonomous import *
from .building import *
from .emulator import *
from .gym import *
from .linear import *
from .nonautonomous import *
from .perturb import *
from .plot import *


systems = {# non-autonomous ODEs
           'CSTR': CSTR,
           'TwoTank': TwoTank,
           # autonomous chaotic ODEs
           'LorenzSystem': LorenzSystem,
           'Lorenz96': Lorenz96,
           'VanDerPol': VanDerPol,
           'ThomasAttractor': ThomasAttractor,
           'RosslerAttractor': RosslerAttractor,
           'LotkaVolterra': LotkaVolterra,
           'Brusselator1D': Brusselator1D,
           'ChuaCircuit': ChuaCircuit,
           'Duffing': Duffing,
           'UniversalOscillator': UniversalOscillator,
           # non-autonomous chaotic ODEs
           'HindmarshRose': HindmarshRose,
           # OpenAI gym environments
           'Pendulum-v0': GymWrapper,
           'CartPole-v1': GymWrapper,
           'Acrobot-v1': GymWrapper,
           'MountainCar-v0': GymWrapper,
           'MountainCarContinuous-v0': GymWrapper,
           # partially observable building state space models with external disturbances
           'SimpleSingleZone': BuildingEnvelope,
           'Reno_full': BuildingEnvelope,
           'Reno_ROM40': BuildingEnvelope,
           'RenoLight_full': BuildingEnvelope,
           'RenoLight_ROM40': BuildingEnvelope,
           'Old_full': BuildingEnvelope,
           'Old_ROM40': BuildingEnvelope,
           'HollandschHuys_full': BuildingEnvelope,
           'HollandschHuys_ROM100': BuildingEnvelope,
           'Infrax_full': BuildingEnvelope,
           'Infrax_ROM100': BuildingEnvelope
           }
