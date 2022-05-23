from psl import plot, systems
from psl.ssm import BuildingEnvelope

import argparse

if __name__ == '__main__':
    """
    Test and plot trajectories of a single system
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('system', choices=[k for k in systems])
    args = parser.parse_args()
    system = systems[args.system]

    if isinstance(system, BuildingEnvelope):
        model = system(system=args.system)
    else:
        model = system()
    out = model.simulate()

    Y = out['Y']
    X = out['X']
    U = out['U'] if 'U' in out.keys() else None
    D = out['D'] if 'D' in out.keys() else None
    plot.pltOL(Y=Y, X=X, U=U, D=D)
    plot.pltPhase(X=Y)
