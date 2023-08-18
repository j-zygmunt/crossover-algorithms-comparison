from dataclasses import dataclass, asdict, field

@dataclass
class GAConfig:
    cx: callable
    cx_params: dict = field(default_factory=dict)
    fun: callable = None
    is_min: bool = True
    dim: int = None
    max_epoch: int = 1000
    pop_size: int = None
    select_size: int = None
    x_max: float = 100.0
    x_min: float = -100.0
    cxpb: float = 0.7
    mutpb: float = 0.1
    elite_size: int = 15


def config_asdict(cfg: GAConfig):
    return {key:(value.__name__ if callable(value) else value) for (key,value) in asdict(cfg).items()}
