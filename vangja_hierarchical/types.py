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
