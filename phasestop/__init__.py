# PhaseStop core public API.
# Import the state machine and its data types from here.
#
# Usage:
#   from phasestop import PhaseStop, Decision, DetectorResult, Signal

from phasestop.config import Decision, DetectorResult, RunResult, Signal
from phasestop.scorer import PhaseStop

__all__ = ["PhaseStop", "Decision", "DetectorResult", "RunResult", "Signal"]
