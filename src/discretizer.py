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

    def get_state_shape(self) -> Tuple[int, ...]:
        return self.state_shape

    def from_dict(config: dict) -> 'Discretizer':
        if not config:
            bins = (5, 5, 20, 20, 15, 20, 2, 2)
            zoom = 1.0
        else:
            bins = config.get('bins', (5, 5, 20, 20, 15, 20, 2, 2))
            zoom = config.get('zoom', 1.0)

            if not isinstance(bins, (list, tuple)):
                raise TypeError(f"'bins' must be a list or tuple, got {type(bins)}")
            if not all(isinstance(b, int) and b > 0 for b in bins):
                raise ValueError(f"'bins' must contain positive integers, got {bins}")
            if not isinstance(zoom, (int, float)) or zoom <= 0:
                raise ValueError(f"'zoom' must be a positive number, got {zoom}")

        return Discretizer(tuple(bins), float(zoom))
