import psl

if __name__ == '__main__':
    """
    Test and plot trajectories of a single system
    """

# choose one key for the system from the dict
print(psl.systems.keys())

name = 'CSTR'
system = psl.systems['CSTR']

if system is psl.BuildingEnvelope:
    ninit = 0
    building = psl.BuildingEnvelope()  # instantiate building class
    building.parameters(system='HollandschHuys_full', linear=False)  # load model parameters
    out = building.simulate(ninit=ninit)
    psl.plot.pltOL(Y=out['Y'], U=out['U'], D=out['D'], X=out['X'])
    psl.plot.pltPhase(X=out['Y'])
elif isinstance(system(), psl.ODE_NonAutonomous):
    model = system(nsim=12000)
    out = model.simulate()  # simulate open loop
    psl.plot.pltOL(Y=out['Y'], U=out['U'])  # plot trajectories
    psl.plot.pltPhase(X=out['Y'])
elif isinstance(system(), psl.ODE_Autonomous):
    model = system()
    out = model.simulate()  # simulate open loop
    psl.plot.pltOL(Y=out['Y'])  # plot trajectories
    psl.plot.pltPhase(X=out['Y'])
