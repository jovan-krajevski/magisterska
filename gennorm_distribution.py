from typing import List, Tuple

import aesara.tensor as at
import numpy as np
import scipy.stats as ss
from aesara.tensor.random.op import RandomVariable
from pymc.aesaraf import floatX
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Continuous
from pymc.distributions.shape_utils import rv_size_is_none


class GenNormRV(RandomVariable):
    name: str = "GenNorm"
    ndim_supp: int = 0
    ndims_params: List[int] = [0, 0, 0]
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("GenNorm", "GGD")

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        beta: np.ndarray,
        loc: np.ndarray,
        scale: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        return ss.gennorm.rvs(beta, loc, scale, random_state=rng, size=size)


gennormrv = GenNormRV()


class GenNorm(Continuous):
    rv_op = gennormrv

    @classmethod
    def dist(cls, beta, loc, scale, *args, **kwargs):
        beta = at.as_tensor_variable(floatX(beta))
        loc = at.as_tensor_variable(floatX(loc))
        scale = at.as_tensor_variable(floatX(scale))
        return super().dist([beta, loc, scale], *args, **kwargs)

    def moment(rv, size, beta, loc, scale):
        moment, _ = at.broadcast_arrays(beta, loc, scale)
        if not rv_size_is_none(size):
            moment = at.full(size, moment)
        return moment

    def logp(value, beta, loc, scale):
        print(beta, loc, scale)
        return check_parameters(
            at.log(beta / (2 * scale)) - at.gammaln(1.0 / beta) -
            (at.abs_(value - loc) / scale)**beta, beta >= 0, scale >= 0)

    def logcdf(value, beta, loc, scale):
        b = value - loc
        c = 0.5 * b / at.abs_(b)
        return (0.5 + c) - c * at.gammaincc(1.0 / beta,
                                            at.abs_(b / scale)**beta)
