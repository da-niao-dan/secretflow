# -*- coding: utf-8 -*-
import logging
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from device_setups import sf_setup_mpc
from performance_stats import time_cost
from utils import (
    Params,
)

import secretflow as sf
from secretflow.device import Device, DeviceObject

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
jax.config.update("jax_enable_x64", True)

def single_round(
    pyu_dev: Device,
    spu_dev: Device,
    params: Params,
    u_i_list: List[DeviceObject],
    M_old: DeviceObject = 0,
    round_num: int = 0,  # useful for benchmark logging
):
    n = len(u_i_list)

    # send to server
    with time_cost("send to server"):
        param_list = [u_i_list[i].to(spu_dev) for i in range(n)]
    
    # https://github.com/encryptogroup/SAFEFL/blob/main/aggregation_rules.py#L208
    # pairwise cosine similarity
    with time_cost("cosine similarity computation"):
        l2_norms = []
        for i in range(n):
            l2_norms.append(spu_dev(lambda x: jnp.sqrt(jnp.sum(jnp.square(x))))(param_list[i]))
        cos_sim_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                ij = spu_dev(lambda a, b: jnp.sum(jnp.multiply(a, b)))(param_list[i], param_list[j])
                d = spu_dev(lambda a, b, c: 1 - jnp.divide(a, jnp.multiply(b, c)))(ij, l2_norms[i], l2_norms[j])
                cos_sim_pairs.append(d)

    # clustering
    # TODO: implement the clustering algorithm and use the result
    with time_cost("clustering"):
        valid_indices = spu_dev(lambda x: x)(jnp.ones(n, dtype=jnp.bool))

    # pairwise euclidean distance
    with time_cost("euclidean norm comparison"):
        euclid_dist = spu_dev(jnp.array)(l2_norms)
        euclid_dist_sorted = spu_dev(lambda x: jnp.sort(x))(euclid_dist)
        clipping_bound = spu_dev(lambda x: x[n//2])(euclid_dist_sorted)

    # gradient clipping
    with time_cost("gradient clipping"):
        gamma = spu_dev(jnp.divide)(clipping_bound, euclid_dist)
        gamma_expanded = spu_dev(jnp.outer)(gamma, jnp.ones(params.m))
        gamma_expanded = spu_dev(jnp.fmin)(gamma_expanded, 1.0)
        gradients = spu_dev(jnp.array)(param_list)
        clipped_gradients = spu_dev(jnp.multiply)(gradients, gamma_expanded)

    # aggregation/noise and update gradients
    with time_cost("aggregation and noise and update"):
        global_update = spu_dev(lambda x: jnp.mean(x, axis=0))(clipped_gradients)
        # TODO: make it adaptive
        noise = jax.random.normal(jax.random.key(0), (params.m,)) * params.sigma
        global_update = spu_dev(jnp.add)(global_update, noise)
        # update the global model
        M_new = spu_dev(jnp.add)(M_old, global_update)
    
    # broadcast the updated model to all clients
    with time_cost("recv from server"):
        sf.reveal(M_new)


def main_mpc(sf_config: dict, party_names, n:int, m: int):
    device_panel, params = sf_setup_mpc(sf_config, party_names, n, m)
    Mt = jnp.zeros((params.m,), dtype=jnp.float64)
    u_i_list = [
        device_panel.client_devices[0](lambda x: x)(jnp.array(np.random.uniform(0, 1, (params.m,))))
        for i in range(n)
    ]
    for i in range(1):
        with time_cost(f"single round {i}"):
            single_round(
                device_panel.client_devices[0], device_panel.server_device, params, u_i_list, M_old=Mt, round_num=i
            )
    sf.shutdown()
