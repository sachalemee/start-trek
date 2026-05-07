##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-16
## File description:
## parse
##

import yaml
from enum import Enum
from typing import Any
from pathlib import Path

class RunMode(Enum):
    TRAIN = "TRAIN"
    EVAL  = "EVAL"

class RenderMode(Enum):
    NONE   = "NONE"
    RECORD = "RECORD"
    HUMAN  = "HUMAN"

SUPPORTED_ALGOS = {"qlearn", "dqn"}

def parse_float(key: str, data: dict, positive: bool) -> float:
    if key in data:
        value = data.get(key)
        if not isinstance(value, float):
            raise TypeError(f"'{key}' must be a float, got {value}")
        elif positive and value < 0:
            raise ValueError(f"'{key}' must be positive, got {value}")
        else:
            return value
    raise KeyError(f"Mandatory '{key}' not set")

def parse_int(key: str, data: dict, positive: bool, mandatory: bool = True) -> int:
    if key in data:
        value = data.get(key)
        if not isinstance(value, int):
            raise TypeError(f"'{key}' must be an int, got {value}")
        elif positive and value < 0:
            raise ValueError(f"'{key}' must be positive, got {value}")
        else:
            return value
    if (mandatory):
        raise KeyError(f"Mandatory '{key}' not set")
    else:
        return 0

def parse_bool(key: str, data: dict) -> bool:
    if key in data:
        value = data.get(key)
        if not isinstance(value, bool):
            raise TypeError(f"'{key}' must be a bool, got {value}")
        else:
            return value
    raise KeyError(f"Mandatory '{key}' not set")

def parse_str(key: str, data: dict) -> str:
    if key in data:
        value = data.get(key)
        if not isinstance(value, str):
            raise TypeError(f"'{key}' must be a str, got {value}")
        else:
            return value
    raise KeyError(f"Mandatory '{key}' not set")

def parse_dict(key: str, data: dict) -> dict:
    if key in data:
        value = data.get(key)
        if not isinstance(value, dict):
            raise TypeError(f"'{key}' must be a dict, got {value}")
        else:
            return value
    raise KeyError(f"Mandatory '{key}' not set")

def require_keys(data: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing required config key: {missing}")

class EnvConfig:
    run_mode: RunMode
    render_mode: RenderMode
    trigger_video: int
    model: str
    nb_loop: int
    max_episode_steps: int
    enable_wind: bool
    wind_power: float
    turbulence_power: float
    mean: int
    mean_steps: int

    def load(self, data: dict):
        raw_run_mode = data.get("run_mode", "TRAIN")
        try:
            run_mode = RunMode(raw_run_mode)
        except ValueError:
            valid = [m.value for m in RunMode]
            raise ValueError(
                f"Invalid run_mode '{raw_run_mode}'. Must be one of: {valid}"
            )

        self.run_mode = run_mode
        raw_render_mode = data.get("render_mode", "NONE")
        try:
            render_mode = RenderMode(raw_render_mode)
        except ValueError:
            valid = [m.value for m in RenderMode]
            raise ValueError(
                f"Invalid render_mode '{raw_render_mode}'. Must be one of: {valid}"
            )

        self.render_mode = render_mode
        self.trigger_video = parse_int("trigger_video", data, True, (self.render_mode == RenderMode.RECORD))
        self.model = parse_str("model", data)
        self.nb_loop = parse_int("nb_loop", data, True)
        self.max_episode_steps = parse_int("max_episode_steps", data, True)
        self.enable_wind = parse_bool("enable_wind", data)
        self.wind_power = parse_float("wind_power", data, True)
        self.turbulence_power = parse_float("turbulence_power", data, True)
        self.mean = parse_int("mean", data, True, (self.render_mode == RunMode.TRAIN))
        self.mean_steps = parse_int("mean_steps", data, True, (self.render_mode == RunMode.TRAIN))

class HyperConfig:
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    discount: float
    lr: float
    batchsize: int
    memory_capacity: int

    def load(self, data: dict, algo: str):
        self.epsilon_start = parse_float("epsilon_start", data, True)
        self.epsilon_end = parse_float("epsilon_end", data, True)
        self.epsilon_decay = parse_float("epsilon_start", data, True)
        self.discount = parse_float("discount", data, True)
        self.lr = parse_float("discount", data, True)
        self.batchsize = parse_int("batchsize", data, True, (algo == "dqn"))
        self.memory_capacity = parse_int("memory_capacity", data, True, (algo == "dqn"))

class ParserConfig:
    env: EnvConfig
    hyper: HyperConfig
    algo: str
    seed: int

    def __init__(self, file: str):
        self.env = EnvConfig()
        self.hyper = HyperConfig()

        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file}")
        with open(path, "r") as f:
            data: Any = yaml.load(f, Loader=yaml.SafeLoader)
        if not isinstance(data, dict):
            raise ValueError("Config file must be a valid YAML")
        require_keys(data, ["env", "hyper", "algo", "seed"])

        algo = parse_str("algo", data)
        if algo not in SUPPORTED_ALGOS:
            raise ValueError(
                f"Unknown algo '{algo}'. Supported: {sorted(SUPPORTED_ALGOS)}"
            )
        self.algo = algo

        self.seed = parse_int("seed", data, False)

        self.env.load(parse_dict("env", data))
        self.hyper.load(parse_dict("hyper", data), self.algo)
