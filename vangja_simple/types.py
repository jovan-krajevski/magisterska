from typing import Literal, TypedDict


class ScaleParams(TypedDict):
    mode: Literal["maxabs", "minmax"]
    y_min: float
    y_max: float
    ds_min: float
    ds_max: float
