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
            X, Y, U, D = building.simulate(ninit=ninit)
            # plot trajectories
            slip.plot.pltOL(Y=Y, U=U, D=D, X=X)
            slip.plot.pltPhase(X=Y)
            plt.close()
        elif isinstance(system(), slip.ODE_NonAutonomous):
            model = system()
            X, Y, U, D = model.simulate()  # simulate open loop
            slip.plot.pltOL(Y=X, U=U)  # plot trajectories
            slip.plot.pltPhase(X=X)
            plt.close()
        elif isinstance(system(), slip.ODE_Autonomous):
            model = system()
            X, Y, U, D = model.simulate()  # simulate open loop
            slip.plot.pltOL(Y=X)  # plot trajectories
            slip.plot.pltPhase(X=X)
            plt.close()

