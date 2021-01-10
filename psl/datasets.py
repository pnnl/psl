import os

resource_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

datasets = {
    k: os.path.join(resource_path, v) for k, v in {
        "tank": "NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat",
        "vehicle3": "NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat",
        "aero": "NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat",
        "flexy_air": "Flexy_air/flexy_air_data.csv",
        "EED_building": "EED_building/EED_building.csv",
        "fsw_phase_1": "FSW/by_step/fsw_data_phase_1.csv",
        "fsw_phase_2": "FSW/by_step/fsw_data_phase_2.csv",
        "fsw_phase_3": "FSW/by_step/fsw_data_phase_3.csv",
        "fsw_phase_4": "FSW/by_step/fsw_data_phase_4.csv",
        "siso_fsw_phase_1": "FSW/siso_by_step/fsw_data_phase_1.csv",
        "siso_fsw_phase_2": "FSW/siso_by_step/fsw_data_phase_2.csv",
        "siso_fsw_phase_3": "FSW/siso_by_step/fsw_data_phase_3.csv",
        "siso_fsw_phase_4": "FSW/siso_by_step/fsw_data_phase_4.csv",
        "siso_nd_fsw_phase_1": "FSW/siso_no_disturb_by_step/fsw_data_phase_1.csv",
        "siso_nd_fsw_phase_2": "FSW/siso_no_disturb_by_step/fsw_data_phase_2.csv",
        "siso_nd_fsw_phase_3": "FSW/siso_no_disturb_by_step/fsw_data_phase_3.csv",
        "siso_nd_fsw_phase_4": "FSW/siso_no_disturb_by_step/fsw_data_phase_4.csv",
    }.items()
}

# systems = {
#     "fsw_phase_1": "datafile",
#     "fsw_phase_2": "datafile",
#     "fsw_phase_3": "datafile",
#     "fsw_phase_4": "datafile",
#     "siso_fsw_phase_1": "datafile",
#     "siso_fsw_phase_2": "datafile",
#     "siso_fsw_phase_3": "datafile",
#     "siso_fsw_phase_4": "datafile",
#     "siso_nd_fsw_phase_1": "datafile",
#     "siso_nd_fsw_phase_2": "datafile",
#     "siso_nd_fsw_phase_3": "datafile",
#     "siso_nd_fsw_phase_4": "datafile",
#     "tank": "datafile",
#     "vehicle3": "datafile",
#     "aero": "datafile",
#     "flexy_air": "datafile",
#     "TwoTank": "emulator",
#     "LorenzSystem": "emulator",
#     "Lorenz96": "emulator",
#     "VanDerPol": "emulator",
#     "ThomasAttractor": "emulator",
#     "RosslerAttractor": "emulator",
#     "LotkaVolterra": "emulator",
#     "Brusselator1D": "emulator",
#     "ChuaCircuit": "emulator",
#     "Duffing": "emulator",
#     "UniversalOscillator": "emulator",
#     "HindmarshRose": "emulator",
#     "SimpleSingleZone": "emulator",
#     "Pendulum-v0": "emulator",
#     "CartPole-v1": "emulator",
#     "Acrobot-v1": "emulator",
#     "MountainCar-v0": "emulator",
#     "MountainCarContinuous-v0": "emulator",
#     "Reno_full": "emulator",
#     "Reno_ROM40": "emulator",
#     "RenoLight_full": "emulator",
#     "RenoLight_ROM40": "emulator",
#     "Old_full": "emulator",
#     "Old_ROM40": "emulator",
#     "HollandschHuys_full": "emulator",
#     "HollandschHuys_ROM100": "emulator",
#     "Infrax_full": "emulator",
#     "Infrax_ROM100": "emulator",
#     "CSTR": "emulator",
#     "UAV3D_kin": "emulator",
#     "UAV2D_kin": "emulator",
# }

train_pid_idxs = [3, 4, 5, 8]
constant_idxs = [6, 7]
train_relay_idxs = [10, 11, 12, 14]
all_train = set(train_pid_idxs + constant_idxs + train_relay_idxs)

all_dev_exp, all_test_exp = [1, 9], [2, 13]
dev_exp, test_exp = [1], [2]

datasplits = {
    "all": {"train": list(all_train), "dev": dev_exp, "test": test_exp},
    "pid": {"train": train_pid_idxs, "dev": dev_exp, "test": test_exp},
    "constant": {"train": constant_idxs, "dev": dev_exp, "test": test_exp},
    "relay": {"train": train_relay_idxs, "dev": dev_exp, "test": test_exp},
    "no_pid": {
        "train": list(all_train - set(train_pid_idxs)),
        "dev": dev_exp,
        "test": test_exp,
    },
    "no_constant": {
        "train": list(all_train - set(constant_idxs)),
        "dev": dev_exp,
        "test": test_exp,
    },
    "no_relay": {
        "train": list(all_train - set(train_relay_idxs)),
        "dev": dev_exp,
        "test": test_exp,
    },
}
