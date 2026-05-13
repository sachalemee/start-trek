##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## termination
##

from enum import Enum

LEFT_LEG  = 6
RIGHT_LEG = 7
OUT_OF_VIEW_X = 1.0

class TerminationCause(Enum):
    NONE        = "none"
    CRASH       = "crash"
    LANDED      = "landed"
    OUT_OF_VIEW = "out_of_view"
    TIMEOUT     = "timeout"

def lander_position(obs) -> tuple[float, float]:
    return float(obs[0]), float(obs[1])

def both_legs_contact(obs) -> bool:
    return bool(obs[LEFT_LEG]) and bool(obs[RIGHT_LEG])

def classify_termination(
    terminated: bool,
    truncated: bool,
    obs,
    reward: float,
) -> TerminationCause:
    if not (terminated or truncated):
        return TerminationCause.NONE

    if terminated:
        both_legs = both_legs_contact(obs)

        if both_legs:
            return TerminationCause.LANDED
        return TerminationCause.CRASH

    x_pos = float(obs[0])
    if abs(x_pos) > OUT_OF_VIEW_X:
        return TerminationCause.OUT_OF_VIEW
    return TerminationCause.TIMEOUT
