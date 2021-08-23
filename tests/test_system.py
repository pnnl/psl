import psl

if __name__ == '__main__':
    """
    Test and plot trajectories of a single system
    """

# choose one key for the system from the dict
print(psl.systems.keys())
name = 'TwoTank'
system = psl.systems[name]

# instantiate selected system model
if system is psl.BuildingEnvelope:
    model = system(system=name)
else:
    model = system()

# simulate open loop
out = model.simulate()

# Plots
Y = out['Y']
X = out['X']
U = out['U'] if 'U' in out.keys() else None
D = out['D'] if 'D' in out.keys() else None
psl.plot.pltOL(Y=Y, X=X, U=U, D=D)
psl.plot.pltPhase(X=Y)
