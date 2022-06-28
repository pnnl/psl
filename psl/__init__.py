import psl.autonomous as auto
import psl.nonautonomous as nauto
import psl.ssm as ssm
import psl.emulator as emulator
import psl.plot as plot

from psl.datasets import datasets

# from psl.autonomous import *
# from psl.nonautonomous import *
# from psl.ssm import *
# from psl.emulator import *
from psl.perturb import Periodic

systems = {**auto.systems, **nauto.systems, **ssm.systems, **emulator.systems}
emulators = systems
