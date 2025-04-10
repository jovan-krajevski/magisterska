from typing import Literal, TypedDict

ScaleMode = Literal["maxabs", "minmax"]


class YScaleParams(TypedDict):
    mode: ScaleMode
    y_min: float
    y_max: float


class TScaleParams(TypedDict):
    ds_min: float
    ds_max: float


Method = Literal[
    "mapx",
    "map",
    "fullrank_advi",
    "advi",
    "svgd",
    "asvgd",
    "nuts",
    "metropolis",
    "demetropolisz",
]

NutsSampler = Literal["pymc", "nutpie", "numpyro", "blackjax"]

FreqStr = Literal[
    "Y",
    "M",
    "W",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us",
    "ns",
    "ps",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
    "picosecond",
]

TuneMethod = Literal["parametric", "prior_from_idata"]

PoolType = Literal["partial", "complete", "individual"]
