# This file implements fig 3 described in https://eprint.iacr.org/2023/206.pdf
from typing import Tuple

import jax
import jax.numpy as jnp
from fss.dcf import DCF_eval, DCF_gen

jax.config.update("jax_enable_x64", True)

# we choose Group G = Z_N, where N = 2 ^ n. We will fix n = 64.

GROUP_BIT_NUM = 64


def share(x, prng_key):
    # x is in Z_N
    # we split x into two shares, x = x0 + x1 mod N
    x0 = jax.random.bits(prng_key, dtype=jnp.uint64)
    x1 = jnp.bitwise_xor(x, x0)
    return x0, x1


def reconstruct(x0, x1):
    return jnp.bitwise_xor(x0, x1)


def DReLU_gen(r_in, r_out, random_key=jax.random.key(1212)):
    random_key_0, random_key_1 = jax.random.split(random_key, 2)
    k0_less, k1_less = DCF_gen(r_in, random_key_0)
    r0, r1 = share(r_out, random_key_1)
    return ([r0] + k0_less, [r1] + k1_less)


def key_to_bytes(key) -> bytes:
    key = jnp.asarray([jnp.uint64(x) for x in key])
    return key.tobytes()


def bytes_to_key(bytes):
    return jnp.frombuffer(bytes, dtype=jnp.uint64)


def DReLU_eval(b, kb, x_hat):
    rb = kb[0]
    kb_less = kb[1:]
    y_hat = x_hat + jnp.pow(jnp.uint64(2), GROUP_BIT_NUM - 1)
    ub = DCF_eval(b, kb_less, x_hat)
    vb = DCF_eval(b, kb_less, y_hat)
    y_hat_b = (
        vb - ub + b * jnp.uint64(y_hat >= jnp.pow(2, GROUP_BIT_NUM - 1)) + rb
    ) & 1
    return y_hat_b


if __name__ == '__main__':
    prng_key = jax.random.PRNGKey(1212)
    k0, k1 = DReLU_gen(2, 3, prng_key)
    k0 = bytes_to_key(key_to_bytes(k0))
    k1 = bytes_to_key(key_to_bytes(k1))
    y0 = DReLU_eval(0, k0, 2)
    y1 = DReLU_eval(1, k1, 2)

    assert reconstruct(y0, y1) == 1

    y0 = DReLU_eval(0, k0, jnp.pow(jnp.uint64(2), GROUP_BIT_NUM - 1) + 1)
    y1 = DReLU_eval(1, k1, jnp.pow(jnp.uint64(2), GROUP_BIT_NUM - 1) + 1)
    assert reconstruct(y0, y1) == 0
