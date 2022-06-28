import psl.autonomous as auto
import psl.nonautonomous as nauto
import psl.ssm as ssm
import psl.emulator as emulator
import psl.plot as plot

systems = {**auto.systems, **nauto.systems, **ssm.systems, **emulator.systems}

