import copy
import logging
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from device_setups import DevicePanel, HandlePanel, sf_setup, sf_setup_prod
from performance_stats import time_cost
from server_program import (
    server_clustering,
)
from utils import (
    Params,
    decrypt_to_jnp_array_gcm,
    encrypt_jnp_array_gcm,
    simulate_data_u,
)

import secretflow as sf
from secretflow.device import Device, DeviceObject

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
jax.config.update("jax_enable_x64", True)

# TODO: single round with pre-computed keys?
def simulate_data_u(
    device_panel: DevicePanel, u_low: float, u_high: float, m: int
) -> Tuple[DeviceObject, DeviceObject]:
    # P_i holds u_i
    uis = [
        device(lambda x: x)(jnp.array(np.random.uniform(u_low, u_high, (m,))))
        for device in device_panel.client_devices
    ]

    return uis

def single_round(
    device_panel: DevicePanel,
    handle_panel: HandlePanel,
    params: Params,
    u_i_list: List[DeviceObject],
    rng: jax.random.PRNGKey,
    M_old: DeviceObject = 0,
    use_tee: bool = False,
    round_num: int = 0,  # useful for benchmark logging
):
    n = len(u_i_list)
    aggregator = device_panel.server_tee if use_tee else device_panel.server_device

    # send to tee
    client_handles = [handle_panel.get_handle(i, -1).to(device_panel.client_devices[i]) for i in range(n)]
    with time_cost("send to tee"):
        # clients encrypt and send the data to server tee
        u_i_list_encrypted = [
            device_panel.client_devices[i](encrypt_jnp_array_gcm)(
                u_i_list[i],
                client_handles[i],
            ).to(device_panel.server_device).to(device_panel.server_tee) for i in range(n)]
        # server tee decrypts the data
        u_i_list_decrypted = [device_panel.server_tee(decrypt_to_jnp_array_gcm)(u_i_list_encrypted[i], handle_panel.get_handle(-1, i)) for i in range(n)]

    # convert to jnp array
    # TODO: rewrite following code for aggregator only
    param_list = [sf.reveal(u_i) for u_i in u_i_list_decrypted]

    # https://github.com/encryptogroup/SAFEFL/blob/main/aggregation_rules.py#L208
    # pairwise cosine similarity
    with time_cost("cosine similarity computation"):
        cos_sim_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d = 1 - jnp.dot(param_list[i], param_list[j]) / (jnp.linalg.norm(param_list[i], ord=2) * jnp.linalg.norm(param_list[j], ord=2))
                cos_sim_pairs.append(d)
        cos_sim_pairs = jnp.array(cos_sim_pairs)

    # clustering
    with time_cost("clustering"):
        valid_indices = server_clustering(
                    device_panel.client_num,
                    cos_sim_pairs,
                    params.eps,
                    params.min_points,
                    params.point_num_threshold,
                    )

        assert len(valid_indices) == device_panel.client_num, (
            f"{len(valid_indices)}, {device_panel.client_num}"
        )

    # pairwise euclidean distance
    with time_cost("euclidean norm comparison"):
        euclid_dist = []
        for grad in param_list:
            euclid_dist.append(jnp.linalg.norm(grad, ord=2))
        clipping_bound = jnp.median(jnp.array(euclid_dist))

    # gradient clipping
    with time_cost("gradient clipping"):
        clipped_gradients = []
        for i in range(device_panel.client_num):
            if valid_indices[i]:
                gamma = clipping_bound / euclid_dist[i]
                clipped_gradients.append(param_list[i] * min(1.0, gamma))

    # aggregation/noise and update gradients
    with time_cost("aggregation and noise and update"):
        global_update = jnp.mean(jnp.stack(clipped_gradients), axis=0)
        # TODO: make it adaptive
        noise = jax.random.normal(rng, (global_update.shape[0],))
        global_update += noise * params.sigma
        # update the global model
        M_new = M_old + global_update
    
    # broadcast the updated model to all clients
    with time_cost("recv from tee"):
        for i in range(n):
            device_panel.server_tee(lambda x: x)(M_new).to(device_panel.server_device).to(device_panel.client_devices[i])

def main():
    device_panel, handle_pannel, params = sf_setup(edge_parties_number=2, m=5)
    rng_key = jax.random.key(0)
    Mt = jnp.zeros((params.m,), dtype=jnp.float64)
    for i in range(1):
        # do training and get u_i_list
        u_i_list = simulate_data_u(device_panel, -1.0, 1.0, params.m)
        # do clustering and get Mt
        with time_cost(f"single round {i}"):
            single_round(
                device_panel, handle_pannel, params, u_i_list, rng_key, Mt, i
            )


def main_prod(sf_config: dict, self_party: str, m: int):
    device_panel, handle_pannel, params = sf_setup_prod(sf_config, self_party, m)
    rng_key = jax.random.key(0)
    Mt = jnp.zeros((params.m,), dtype=jnp.float64)
    for i in range(1):
        # do training and get u_i_list
        u_i_list = simulate_data_u(device_panel, -1.0, 1.0, params.m)
        # do clustering and get Mt
        with time_cost(f"single round {i}"):
            single_round(
                device_panel, handle_pannel, params, u_i_list, rng_key, M_old=Mt, round_num=i
            )
    sf.shutdown()


if __name__ == "__main__":
    main()
