import jax
import jax.numpy as jnp
from fss.drelu import (
    GROUP_BIT_NUM,
    DReLU_eval,
    DReLU_gen,
    bytes_to_key,
    key_to_bytes,
    reconstruct,
)
from utils import decrypt_gcm, encrypt_gcm, get_random_bytes

if __name__ == '__main__':
    prng_key = jax.random.PRNGKey(1212)
    aes_key = get_random_bytes(32)
    k0, k1 = DReLU_gen(2, 3, prng_key)
    k0 = key_to_bytes(k0)
    k1 = key_to_bytes(k1)

    k0_enc = encrypt_gcm(k0, aes_key)
    k1_enc = encrypt_gcm(k1, aes_key)
    k0 = bytes_to_key(decrypt_gcm(k0_enc[0], k0_enc[1], k0_enc[2], aes_key))
    k1 = bytes_to_key(decrypt_gcm(k1_enc[0], k1_enc[1], k1_enc[2], aes_key))

    y0 = DReLU_eval(0, k0, 2)
    y1 = DReLU_eval(1, k1, 2)

    assert reconstruct(y0, y1) == 1

    y0 = DReLU_eval(0, k0, jnp.pow(jnp.uint64(2), GROUP_BIT_NUM - 1) + 1)
    y1 = DReLU_eval(1, k1, jnp.pow(jnp.uint64(2), GROUP_BIT_NUM - 1) + 1)
    assert reconstruct(y0, y1) == 0
    print("successfully passed test")
