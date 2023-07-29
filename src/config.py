from dataclasses import dataclass, asdict, field

@dataclass
class GAConfig:
    cx: callable
    cx_params: dict = field(default_factory=dict)
    fun: callable = None
    is_min: bool = True
    dim: int = 2
    max_epoch: int = 100
    pop_size: int = 500
    x_max: float = 100.0
    x_min: float = -100.0
    cxpb: float = 0.8
    mutpb: float = 0.2
    elite_size: int = 5


def config_asdict(cfg: GAConfig):
    return {key:(value.__name__ if callable(value) else value) for (key,value) in asdict(cfg).items()}
