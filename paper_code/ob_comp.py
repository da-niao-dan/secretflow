import jax
import jax.numpy as jnp
from utils import (
    Devices,
    Handles,
    Params,
    corr,
    decrypt_to_jnp_array_gcm,
    encrypt_jnp_array_gcm,
    gen_handles,
    simulate_data_u,
)
from fss.drelu import DReLU_gen, DReLU_eval

import secretflow as sf
from secretflow.device import DeviceObject

jax.config.update("jax_enable_x64", True)


def L2(x):
    return jnp.linalg.norm(x, ord=2)


def make_random_shares(rng: jax.random.PRNGKey, dtype=jnp.uint64):
    """
    Generate random shares of a given shape and dtype.
    """
    rng1, rng2 = jax.random.split(rng, 2)
    return jax.random.bits(rng1, dtype), jax.random.bits(rng2, dtype)


def preprcessing(devices: Devices, rng: jax.random.PRNGKey):
    r0, r1 = devices.server_tee(make_random_shares)(rng)
    devices.server_tee(DReLU_gen(r0,r1))

def ob_comp(
    L_i,
    L_j,
    devices: Devices,
    handles: Handles,
    params: Params,
    seed: int = 1212,
    verbose=False,
) -> DeviceObject:
    return
