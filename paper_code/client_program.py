from typing import List

import jax
import jax.numpy as jnp
from fss.drelu import DReLU_eval
from utils import decrypt_to_jnp_array_gcm, encrypt_jnp_array_gcm

from secretflow.device import DeviceObject

jax.config.update("jax_enable_x64", True)


# key unpacking
def key_unpack(ci, handle_i_s):
    decrypted_bytes = decrypt_to_jnp_array_gcm(ci, handle_i_s)
    key = decrypted_bytes[:-1]
    r = decrypted_bytes[-1]
    return key, r


# Euclidean distance comparison with each E𝑗
def encode_c_Lij_i(Li, ri, handle_i_j, is_j=False):
    if is_j:
        Lij_i = -Li - ri
    else:
        Lij_i = Li - ri
    c_Lij_i = encrypt_jnp_array_gcm(Lij_i, handle_i_j)
    return c_Lij_i


def compute_Lij(c_Lij_j: DeviceObject, Lij_i: DeviceObject, handle_i_j):
    Lij_j = decrypt_to_jnp_array_gcm(c_Lij_j, handle_i_j)
    Lij = Lij_j + Lij_i
    return Lij


def compute_zij_i(party_indicator, FKeyi, Lij, edge_tee_i):
    zij_i = edge_tee_i(DReLU_eval)(party_indicator, FKeyi, Lij)
    return zij_i


def compute_c_zij_i(zij_i, b_i, device_i, handle_i_s):

    zij_i = device_i(lambda x, y: jnp.bitwise_xor(x, y))(zij_i, b_i)
    c_zij_i = device_i(lambda z, key: encrypt_jnp_array_gcm(z, key))(zij_i, handle_i_s)
    return c_zij_i


# receive meta information and clipping
def compute_c_h(i, c_I, c_med, handle_i_s) -> List:
    I = decrypt_to_jnp_array_gcm(c_I, handle_i_s)
    med = decrypt_to_jnp_array_gcm(c_med, handle_i_s)
    c_h_list = []
    assert i == med, "med computed is not the same with i"
    for h in I:
        c_h = encrypt_jnp_array_gcm(h, handle_i_s)
        c_h_list.append(c_h)
    return c_h_list


def clipping(i, is_accepted, c_I, c_L, u_i, handle_i_s, handle_i_med):
    L_i = jnp.linalg.norm(u_i, ord=2)
    I = decrypt_to_jnp_array_gcm(c_I, handle_i_s)
    if is_accepted:
        L_med = decrypt_to_jnp_array_gcm(c_L, handle_i_med)
        beta = L_med / L_i
        u_i_F = u_i * min(1, beta)
        return u_i_F
    assert I[i] == 0, "the server mistaken me! I should be valid"
    return None


# local filtering and aggregation if Pi's index i in I^C


def smallest_noise(u_i_F, sigma, rng_key):
    noise = jax.random.normal(rng_key, (u_i_F.shape[0],))
    return u_i_F + sigma * noise


def compute_c_Ut_i(rij_sum_list, rij_subtract_list, u_i_F, handle_i_s):
    Ut_i = u_i_F + jnp.sum(rij_sum_list) - jnp.sum(rij_subtract_list)
    c_Ut_i = encrypt_jnp_array_gcm(Ut_i, handle_i_s)
    return c_Ut_i
