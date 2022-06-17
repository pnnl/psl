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
        "9bus_test": "9bus_test",
        "9bus_init": "9bus_perturbed_init_cond",
        "200bus_boundary": "ACTIVSg200_case141_data/dyn_202.csv",
        "real_linear_xva": "Real_Linear_xva/Real_Linear_xva.csv",
        "pendulum_h_1": "Ideal_Pendulum/Ideal_Pendulum.csv"
    }.items()
}
