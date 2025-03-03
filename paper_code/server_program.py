from collections import Counter
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from fss.drelu import DReLU_eval, DReLU_gen, bytes_to_key, key_to_bytes
from sklearn.cluster import DBSCAN
from utils import (
    Devices,
    Handles,
    Params,
    corr,
    corr_rand_distribute,
    decrypt_gcm,
    decrypt_to_jnp_array_gcm,
    encrypt_gcm,
    encrypt_jnp_array_gcm,
    gen_handles,
    make_random_shares,
    simulate_data_u,
)

from secretflow.device import Device, DeviceObject

jax.config.update("jax_enable_x64", True)


def initialization():
    pass


def preprocessing(
    rng: jax.random.PRNGKey,
    handles: Handles,
):

    r0, r1 = make_random_shares(rng)
    r = r0 + r1
    fkey_i, fkey_j = DReLU_gen(r, 0)
    fkey_i_with_r0 = fkey_i + [r0]
    fkey_j_with_r1 = fkey_j + [r1]

    c_i = encrypt_gcm(key_to_bytes(fkey_i_with_r0), handles.s_i)
    c_j = encrypt_gcm(key_to_bytes(fkey_j_with_r1), handles.s_j)
    return c_i, c_j


def server_reconstruct_cosij(
    c,
    c_cosij_i,
    c_cosij_j,
    cos_shape,
    devices: Devices,
    handles: Handles,
    params: Params,
):

    cosij_i = devices.server_tee(
        lambda c, gcm_key: decrypt_to_jnp_array_gcm(c, gcm_key), num_returns=1
    )(c_cosij_i[0], c_cosij_i[1], c_cosij_i[2], handles.s_i, params.fxp_type, cos_shape)

    cosij_j = devices.server_tee(
        lambda c, gcm_key: decrypt_to_jnp_array_gcm(c, gcm_key), num_returns=1
    )(c_cosij_j[0], c_cosij_j[1], c_cosij_j[2], handles.s_j, params.fxp_type, cos_shape)

    cosij = devices.server_tee(
        lambda cosij_i, cosij_j, c: cosij_i + cosij_j + jnp.sum(c), num_returns=1
    )(cosij_i, cosij_j, c)

    return cosij


def server_clustering(
    n: int, cosij_list: List, epsilon: float, minPts: int, outlier_threshold: int
) -> np.ndarray:
    """server clustering the cos sim pairs to determine which points are outliners

    Args:
        n (int): number of points
        cosij_list (List): pairwise cos sim between n vectors, ordered.
        epsilon (float): smaller than which two cos values are considered close.
        minPts (int): minimum number of points to be considered as a cluster
        outlier_threshold (int): Points are considered outliers,
            if their appearance count in abnormal pairs is
            larger than this specified threshold.

    Returns:
        np.ndarray: valid indices of points that are not outliers, 1 is valid.
    """
    assert (n * (n - 1) / 2) == len(cosij_list)
    X = np.array(cosij_list).reshape(-1, 1)
    db = DBSCAN(eps=epsilon, min_samples=minPts).fit(X)
    labels = db.labels_
    # return the index of labels > -1
    outlier_pair_indices = np.where(labels == -1)[0]
    outlier_counter = np.zeros(n)
    i = 0
    for i in range(n):
        for j in range(i + 1, n):
            if outlier_pair_indices[i] == -1:
                outlier_counter[i] += 1
                outlier_counter[j] += 1
            i += 1
    valid_indices = np.where(outlier_counter < outlier_threshold)[0]

    return valid_indices.astype(np.int32)


def index_encode(valid_indices, server_tee, handle_s_h):
    c_I = server_tee(
        lambda x, gcm_key: encrypt_jnp_array_gcm(x, gcm_key), num_returns=1
    )(valid_indices, handle_s_h)
    return c_I


# Median index determination of euclidean distances


def reconstruct_zij(c_zij_i, c_zij_j, devices: Devices, handles: Handles):
    zij_i = devices.server_tee(
        lambda c, gcm_key: decrypt_to_jnp_array_gcm(c, gcm_key), num_returns=1
    )(c_zij_i, handles.s_i)
    zij_j = devices.server_tee(
        lambda c, gcm_key: decrypt_to_jnp_array_gcm(c, gcm_key), num_returns=1
    )(c_zij_j, handles.s_j)
    zij = devices.server_tee(lambda x, y: jnp.bitwise_xor(x, y))(zij_i, zij_j)
    return zij


def median_and_index(zij_list, server_tee):
    median = server_tee(lambda x: jnp.median(x))(zij_list)
    median_index = server_tee(lambda z: jnp.where(z == median)[0][0], num_returns=1)(
        zij_list
    )
    return median, median_index


def median_index_encode(median_index, handle_s_i_list, server_tee: DeviceObject):
    handle_s_med = server_tee(lambda l, index: l[index])(handle_s_i_list, median_index)
    c_median_index = server_tee(
        lambda x, gcm_key: encrypt_jnp_array_gcm(x, gcm_key), num_returns=1
    )(median_index, handle_s_med)
    return c_median_index


# note the paper said send encoded valid indices to device h, chv is sent, not dealing with it here.
def package_information(h: int, valid_indices: np.ndarray, chL):
    vh = valid_indices[h]
    if vh == 1:
        # 1 means accepted
        return 1, chL
    else:
        # 0 means rejected
        return 0, None


# aggregation
def aggregation(
    c_ut_list: list[DeviceObject],
    M_t_1: DeviceObject,
    server_tee: Device,
    handle_s_h_list: List[DeviceObject],
    handle_s_i_list: List[DeviceObject],  # broadcast to all devices
    params: Params,
):
    ut_list = []
    for c_ut, handle_s_h in zip(c_ut_list, handle_s_h_list):
        ut = server_tee(lambda c, key: decrypt_to_jnp_array_gcm(c, key))(
            c_ut, handle_s_h, params.fxp_type, (-1)
        )
        ut_list.append(ut)
    Mt = server_tee(lambda ut_list, M_t_1: jnp.mean(ut_list) + M_t_1)(ut_list, M_t_1)

    ch_Mt_list = []
    for handle_s_h in handle_s_i_list:
        ch_Mt = server_tee(lambda Mt, key: encrypt_jnp_array_gcm(Mt, key))(
            Mt, handle_s_h
        )
        ch_Mt_list.append(ch_Mt)

    return Mt, ch_Mt_list


if __name__ == "__main__":
    mean = -0.5
    std_dev = 0.1
    num_samples = 100
    samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
    samples[0] = 0
    samples[1] = 1
    print(server_clustering(samples, 0.1, 2))
