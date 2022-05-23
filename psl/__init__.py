import psl.autonomous as auto
import psl.nonautonomous as nauto
import psl.ssm as ssm
import psl.emulator as emulator

systems = {**auto.systems, **nauto.systems, **ssm.systems, **emulator.systems}