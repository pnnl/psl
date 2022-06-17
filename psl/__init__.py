from .autonomous import *
from .emulator import *
from .nonautonomous import *
from .datasets import *
from .perturb import *
from .plot import *

emulators = {
    # non-autonomous ODEs
    "CSTR": CSTR,
    "Tank": Tank,
    "TwoTank": TwoTank,
    "InvPendulum": InvPendulum,
    "UAV3D_kin": UAV3D_kin,
    "UAV2D_kin": UAV2D_kin,
    "UAV3D_reduced": UAV3D_reduced,
    "SEIR_population": SEIR_population,
    "Iver_kin_reduced": Iver_kin_reduced,
    "Iver_kin": Iver_kin,
    "Iver_dyn": Iver_dyn,
    "Iver_dyn_reduced": Iver_dyn_reduced,
    "Iver_dyn_simplified": Iver_dyn_simplified,
    "Iver_dyn_simplified_output": Iver_dyn_simplified_output,

    # autonomous chaotic ODEs
    "UniversalOscillator": UniversalOscillator,
    "LorenzSystem": LorenzSystem,
    "Pendulum": Pendulum,
    "Lorenz96": Lorenz96,
    "VanDerPol": VanDerPol,
    "ThomasAttractor": ThomasAttractor,
    "RosslerAttractor": RosslerAttractor,
    "LotkaVolterra": LotkaVolterra,
    "Brusselator1D": Brusselator1D,
    "ChuaCircuit": ChuaCircuit,
    "Duffing": Duffing,
    "DoublePendulum": DoublePendulum,
    "Autoignition": Autoignition,

    # non-autonomous chaotic ODEs
    "HindmarshRose": HindmarshRose,

    # OpenAI gym environments
    "Pendulum-v0": GymWrapper,
    "CartPole-v1": GymWrapper,
    "Acrobot-v1": GymWrapper,
    "MountainCar-v0": GymWrapper,
    "MountainCarContinuous-v0": GymWrapper,

    # partially observable building state space models with external disturbances
    "SimpleSingleZone": BuildingEnvelope,
    "Reno_full": BuildingEnvelope,
    "Reno_ROM40": BuildingEnvelope,
    "RenoLight_full": BuildingEnvelope,
    "RenoLight_ROM40": BuildingEnvelope,
    "Old_full": BuildingEnvelope,
    "Old_ROM40": BuildingEnvelope,
    "HollandschHuys_full": BuildingEnvelope,
    "HollandschHuys_ROM100": BuildingEnvelope,
    "Infrax_full": BuildingEnvelope,
    "Infrax_ROM100": BuildingEnvelope,

    # Power grid
    "SwingEquation": SwingEquation,
}

systems = emulators