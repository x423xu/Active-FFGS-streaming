from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]

    def as_float(self):
        return Gaussians(
            means=self.means.float(),
            covariances=self.covariances.float(),
            harmonics=self.harmonics.float(),
            opacities=self.opacities.float(),
        )
