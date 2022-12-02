from psl import plot
import matplotlib.pyplot as plt
import os

import psl.autonomous as auto
import psl.nonautonomous as nauto
import psl.ssm as ssm
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    """
    Tests
    """
    os.system('rm -rf figs')
    os.mkdir("./figs")
    ninit = 0

    for name, system in ssm.systems.items():
        print(name)
        ssm = system(system=name)
        out = ssm.simulate(ninit=ninit)
        plot.pltOL(Y=out['Y'], U=out['U'], D=out['D'], X=out['X'], figname="./figs/" + name + "_ol")
        plot.pltPhase(X=out['Y'], figname="./figs/" + name + "_phase")
        plt.close('all')

    for name, system in nauto.systems.items():
        print(name)
        model = system(nsim=1000)
        out = model.simulate()
        plot.pltOL(Y=out['Y'], U=out['U'], figname="./figs/"+name+"_ol")  # plot trajectories
        plot.pltPhase(X=out['Y'], figname="./figs/"+name+"_phase")
        plt.close('all')

    for name, system in auto.systems.items():
        print(name)
        model = system()
        out = model.simulate()
        plot.pltOL(Y=out['Y'], figname="./figs/"+name+"_ol")  # plot trajectories
        plot.pltPhase(X=out['Y'], figname="./figs/"+name+"_phase")
        plt.close('all')

