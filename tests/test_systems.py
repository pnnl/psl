import psl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    """
    Tests
    """

    for name, system in psl.systems.items():
        print(name)
        if system is psl.BuildingEnvelope:
            ninit = 0
            building = psl.BuildingEnvelope()  # instantiate building class
            building.parameters(system='HollandschHuys_full', linear=False)  # load model parameters
            # simulate open loop building
            out = building.simulate(ninit=ninit)
            # plot trajectories
            psl.plot.pltOL(Y=out['Y'], U=out['U'], D=out['D'], X=out['X'])
            psl.plot.pltPhase(X=out['Y'])
            plt.close('all')
        elif isinstance(system(), psl.ODE_NonAutonomous):
            model = system(nsim=12000)
            out = model.simulate()  # simulate open loop
            psl.plot.pltOL(Y=out['Y'], U=out['U'])  # plot trajectories
            psl.plot.pltPhase(X=out['Y'])
            plt.close('all')
        elif isinstance(system(), psl.ODE_Autonomous):
            model = system()
            out = model.simulate()  # simulate open loop
            psl.plot.pltOL(Y=out['Y'])  # plot trajectories
            psl.plot.pltPhase(X=out['Y'])
            plt.close('all')

