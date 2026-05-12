##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## discretizer
##

import numpy as np
from typing import Tuple, List

class Discretizer:

    def __init__(self, bins: Tuple[int, ...], zoom: float = 1.0):
        self.bins = bins # bins par dimension
        self.zoom = zoom
        self.state_shape = bins # forme de la qtable

    def discretize(self, obs: List[float]) -> Tuple[int, ...]:
        return tuple([round(e * self.zoom) for e in obs]) # transforme l'obs contunue en indices discrets et crée un tuple pour accéder à la qtable
