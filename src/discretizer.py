##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## discretizer
##

import numpy as np
from typing import Tuple, List

baseBins = (5, 5, 20, 20, 15, 20, 2, 2)

class Discretizer:

    def __init__(self, bins: Tuple[int, ...], zoom: float = 1.0):
        self.bins = bins
        self.zoom = zoom

    def discretize(self, obs: List[float]) -> Tuple[int, ...]:
        tmp = ()
        for i in range(len(obs)):
            tmp += (round((obs[i] * (self.bins[i] / baseBins[i])) * self.zoom),)
        return tmp

    def get_state_shape(self) -> Tuple[int, ...]:
        return self.bins

    def from_dict(config: dict) -> 'Discretizer':
        if not config:
            bins = baseBins
            zoom = 1.0
        else:
            bins = config.get('bins', baseBins)
            zoom = config.get('zoom', 1.0)

            if not isinstance(bins, (list, tuple)):
                raise TypeError(f"'bins' must be a list or tuple, got {type(bins)}")
            if not all(isinstance(b, int) and b > 0 for b in bins):
                raise ValueError(f"'bins' must contain positive integers, got {bins}")
            if not isinstance(zoom, (int, float)) or zoom <= 0:
                raise ValueError(f"'zoom' must be a positive number, got {zoom}")

        return Discretizer(tuple(bins), float(zoom))
