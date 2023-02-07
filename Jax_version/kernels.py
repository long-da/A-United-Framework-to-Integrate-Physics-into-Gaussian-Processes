import jax.numpy as jnp
from jax import grad, jit
from functools import partial
from jax.config import config
import math

config.update("jax_enable_x64", True)


class RBF_kernel_u_1d(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, y1, s1):
        return (jnp.exp(-1 / 2 * ((x1 - y1)**2 / s1**2))).sum()

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, y1, s1):
        val = grad(self.kappa, 0)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, y1, s1):
        val = grad(grad(self.kappa, 0), 0)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, y1, s1):
        val = grad(self.kappa, 1)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, y1, s1):
        val = grad(grad(self.kappa, 1), 1)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, y1, s1):
        val = grad(grad(self.kappa, 0), 1)(x1, y1, s1)
        return val
    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, y1, s1):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 1),1)(x1, y1, s1)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y1_kappa(self, x1, y1, s1):
        val = grad(grad(grad(self.kappa, 0), 1), 1)(x1, y1, s1)
        return val


class RBF_kernel_u(object):

    def __init__(self):
        pass

    @partial(jit, static_argnums=(0, ))
    def kappa(self, x1, x2, y1, y2, s1, s2):
        return jnp.exp(-1 / 2 * ((x1 - y1)**2 / s1**2 + (x2 - y2)**2 / s2**2))

    @partial(jit, static_argnums=(0, ))
    def D_x1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(self.kappa, 0)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(self.kappa, 1)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 0), 0)(x1, x2, y1, y2, s1, s2)
        return val


    @partial(jit, static_argnums=(0, ))
    def DD_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 2), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 1), 1)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(self.kappa, 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(self.kappa, 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 0), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 0), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x2_D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 1), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x2_D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(self.kappa, 1), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x1_DD_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 0), 3), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def D_x2_DD_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 1), 3), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x2_DD_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(grad(self.kappa, 1), 1), 3), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 2), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 0), 0), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 0), 0), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x2_D_y1_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 1), 1), 2)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x2_D_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(self.kappa, 1), 1), 3)(x1, x2, y1, y2, s1, s2)
        return val

    @partial(jit, static_argnums=(0, ))
    def DD_x1_DD_y2_kappa(self, x1, x2, y1, y2, s1, s2):
        val = grad(grad(grad(grad(self.kappa, 0), 0), 3), 3)(x1, x2, y1, y2, s1, s2)
        return val