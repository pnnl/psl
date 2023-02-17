import psl.autonomous as auto
import psl.nonautonomous as nauto
import psl.ssm as ssm
import psl.coupled_systems as cs
import psl.emulator as emulator
import psl.plot as plot
from psl.perturb import *

import os

resource_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

datasets = {
    k: os.path.join(resource_path, v) for k, v in {
        "tank": "NLIN_SISO_two_tank/NLIN_two_tank_SISO.mat",
        "vehicle3": "NLIN_MIMO_vehicle/NLIN_MIMO_vehicle3.mat",
        "aero": "NLIN_MIMO_Aerodynamic/NLIN_MIMO_Aerodynamic.mat",
        "flexy_air": "Flexy_air/flexy_air_data.csv",
        "EED_building": "EED_building/EED_building.csv",
        "9bus_test": "9bus_test",
        "9bus_init": "9bus_perturbed_init_cond",
        "real_linear_xva": "Real_Linear_xva/Real_Linear_xva.csv",
        "pendulum_h_1": "Ideal_Pendulum/Ideal_Pendulum.csv"
    }.items()
}

systems = {**auto.systems, **nauto.systems, **ssm.systems, **cs.systems}
emulators = systems

