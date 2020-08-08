import slip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    """
    Tests
    """

    for name, system in slip.systems.items():
        print(name)
        if system is slip.BuildingEnvelope:
            ninit = 0
            building = slip.BuildingEnvelope()  # instantiate building class
            building.parameters(system='HollandschHuys_full', linear=False)  # load model parameters
            # simulate open loop building
            out = building.simulate(ninit=ninit)
            # plot trajectories
            slip.plot.pltOL(Y=out['Y'], U=out['U'], D=out['D'], X=out['X'])
            slip.plot.pltPhase(X=out['Y'])
            plt.close('all')
        elif isinstance(system(), slip.ODE_NonAutonomous):
            model = system()
            out = model.simulate()  # simulate open loop
            slip.plot.pltOL(Y=out['Y'], U=out['U'])  # plot trajectories
            slip.plot.pltPhase(X=out['Y'])
            plt.close('all')
        elif isinstance(system(), slip.ODE_Autonomous):
            model = system()
            out = model.simulate()  # simulate open loop
            slip.plot.pltOL(Y=out['Y'])  # plot trajectories
            slip.plot.pltPhase(X=out['Y'])
            plt.close('all')

