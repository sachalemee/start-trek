##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## termination
##

from enum import Enum

LEFT_LEG  = 6
RIGHT_LEG = 7
VELOCITY_THRESHOLD = 0.1
CRASH_REWARD_THRESHOLD = -50.0
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

def is_settled(obs) -> bool:
    # linear/angular speeds close to zero -> motor off/idling
    return (
        abs(float(obs[2])) < VELOCITY_THRESHOLD and
        abs(float(obs[3])) < VELOCITY_THRESHOLD and
        abs(float(obs[5])) < VELOCITY_THRESHOLD
        )

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
        settled = is_settled(obs)
        not_crash = reward > CRASH_REWARD_THRESHOLD

        if both_legs and settled and not_crash:
            return TerminationCause.LANDED
        return TerminationCause.CRASH

    x_pos = float(obs[0])
    if abs(x_pos) > OUT_OF_VIEW_X:
        return TerminationCause.OUT_OF_VIEW
    return TerminationCause.TIMEOUT
