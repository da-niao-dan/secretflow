import logging
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from client_program import (
    clipping,
    compute_c_h_L,
    compute_c_Ut_i,
    compute_c_zij_i,
    compute_Lij,
    compute_zij_i,
    encode_c_Lij_i,
    key_unpack,
    smallest_noise,
)
from cos_sim import cos_sim
from device_setups import DevicePanel, HandlePanel, sf_setup, sf_setup_prod
from performance_stats import time_cost
from server_program import (
    aggregation,
    index_encode,
    median_and_index,
    median_index_encode,
    package_information,
    preprocessing,
    reconstruct_zij,
    server_clustering,
)
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

import secretflow as sf
from secretflow.device import Device, DeviceObject

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
jax.config.update("jax_enable_x64", True)


def key_gen(
    device_panel: DevicePanel,
    handle_panel: HandlePanel,
    random_key: jax.random.PRNGKey,
):
    """Generatie FSS keys for each pair of edge devices"""
    device_FKeys = {}
    for i, j in device_panel.enumerate_pairs():
        devices = device_panel.build_devices(i, j)
        handles = handle_panel.build_handles(i, j)
        random_key_use, random_key = jax.random.split(random_key, 2)
        c_i, c_j = device_panel.server_tee(preprocessing, num_returns=2)(
            random_key_use, handles.s_i, handles.s_j
        )
        # give keys to devices
        device_FKeys[(i, j)] = (
            c_i.to(devices.edge_device_i).to(devices.edge_tee_i),
            c_j.to(devices.edge_device_j).to(devices.edge_tee_j),
        )
    return device_FKeys


# FL program steps functions
def corr_rand_distr_pairwise(
    device_panel: DevicePanel, handle_panel: HandlePanel, params: Params
):
    abs_pairs = {}
    for i, j in device_panel.enumerate_pairs():
        devices = device_panel.build_devices(i, j)
        handles = handle_panel.build_handles(i, j)
        server_a, server_b, c, edge_tee_i_a, edge_tee_j_b = corr_rand_distribute(
            devices, handles, params
        )
        abs_pairs[(i, j)] = (server_a, server_b, c, edge_tee_i_a, edge_tee_j_b)
    return abs_pairs


def cos_sim_pairwise(
    u_i_list_tee: List[DeviceObject],
    abc_info: Dict[Tuple, Tuple[DeviceObject]],
    device_panel: DevicePanel,
    handle_panel: HandlePanel,
    params: Params,
):
    cos_sim_pairs = []
    for i, j in device_panel.enumerate_pairs():
        devices = device_panel.build_devices(i, j)
        handles = handle_panel.build_handles(i, j)
        cos_sim_i_j = cos_sim(
            u_i_list_tee[i],
            u_i_list_tee[j],
            devices,
            handles,
            params,
            abc_info.get((i, j)),
        )
        # cos_sim_i_j is at server tee
        cos_sim_pairs.append(cos_sim_i_j)
    return cos_sim_pairs


def euclidean_norm_pairwise(
    u_i_list_tee: List[DeviceObject],
    device_FKeys: Dict[Tuple, Tuple],
    device_panel: DevicePanel,
    handle_panel: HandlePanel,
    params: Params,
) -> Dict[Tuple, Tuple[DeviceObject, DeviceObject]]:
    # use fixed point encoding
    norm_i_list_tee = [
        u_i.device(lambda x: params.fxp_type(jnp.linalg.norm(x) * (2.0**params.fxp)))(
            u_i
        )
        for u_i in u_i_list_tee
    ]
    c_zij_dict = {}
    for i, j in device_panel.enumerate_pairs():
        devices = device_panel.build_devices(i, j)
        handles = handle_panel.build_handles(i, j)

        c_i, c_j = device_FKeys[(i, j)]
        key_i, r_i = devices.edge_tee_i(key_unpack, num_returns=2)(c_i, handles.i_s)
        key_j, r_j = devices.edge_tee_j(key_unpack, num_returns=2)(c_j, handles.j_s)
        c_normij_i = devices.edge_device_i(encode_c_Lij_i)(
            norm_i_list_tee[i], r_i, handles.i_j
        )
        # checkout Fig 4:  Program installed by client P𝑖
        c_normij_j = devices.edge_device_j(encode_c_Lij_i)(
            norm_i_list_tee[j], r_j, handles.j_i, True
        )

        norm_ij_i = devices.edge_tee_i(compute_Lij)(
            c_normij_j.to(devices.server_device).to(devices.edge_tee_i),
            norm_i_list_tee[i],
            handles.i_j,
        )
        norm_ij_j = devices.edge_tee_j(compute_Lij)(
            c_normij_i.to(devices.server_device).to(devices.edge_tee_j),
            norm_i_list_tee[j],
            handles.j_i,
        )
        zij_i = compute_zij_i(0, key_i, norm_ij_i, devices.edge_tee_i)
        zij_j = compute_zij_i(1, key_j, norm_ij_j, devices.edge_tee_j)
        b_i, b_j = corr(
            params.k,
            1,
            devices.edge_tee_i,
            handles.i,
            devices.edge_tee_j,
            handles.j,
            return_zero_sharing=True,
        )

        c_zij_i = (
            compute_c_zij_i(zij_i, b_i, devices.edge_tee_i, handles.i_s)
            .to(devices.edge_device_i)
            .to(devices.server_tee)
        )
        c_zij_j = (
            compute_c_zij_i(zij_j, b_j, devices.edge_tee_j, handles.j_s)
            .to(devices.edge_device_j)
            .to(devices.server_tee)
        )
        c_zij_dict[(i, j)] = (c_zij_i, c_zij_j)
    return c_zij_dict


def valid_index_encode_devicewise(
    devices_panel: DevicePanel, handle_panel: HandlePanel, index_list: List
) -> List[DeviceObject]:
    index_list_at_clients = [
        devices_panel.server_tee(index_encode)(index_list, handle_s_h).to(tee)
        for tee, handle_s_h in [
            *zip(devices_panel.client_tees, handle_panel.get_server_handles())
        ]
    ]
    return index_list_at_clients


def reconstruct_zij_pairwise(
    device_panel: DevicePanel,
    handle_panel: HandlePanel,
    c_zij_dict: Dict[Tuple, Tuple[DeviceObject, DeviceObject]],
    params: Params,
):
    zij_list = []
    for i, j in device_panel.enumerate_pairs():
        devices = device_panel.build_devices(i, j)
        handles = handle_panel.build_handles(i, j)
        c_zij_i, c_zij_j = c_zij_dict[(i, j)]
        zij_list.append(
            devices.server_tee(reconstruct_zij)(
                c_zij_i, c_zij_j, handles.s_i, handles.s_j, params.fxp
            )
        )
    return zij_list


# receive meta info and clipping
def receive_meta_info_and_clipping(
    valid_index_at_clients,
    valid_indices,
    med_encoded,
    median_index_server_tee,
    device_panel: DevicePanel,
    handle_panel: HandlePanel,
    params: Params,
    u_i_list: List[DeviceObject],
):
    # sends to Pmed
    median_index = sf.reveal(median_index_server_tee)
    Pmed = device_panel.client_devices[median_index]
    # median device already has valid_index in valid_index_at_clients
    median_encoded_Pmed = med_encoded.to(Pmed)
    # give all to Emed
    Emed = device_panel.client_tees[median_index]
    valid_index_at_Emed = valid_index_at_clients[median_index].to(Emed)
    median_encoded_Emed = median_encoded_Pmed.to(Emed)

    # TODO: rewrite this
    c_h_L = Emed(compute_c_h_L)(
        median_index,
        device_panel.client_num,
        valid_index_at_Emed,
        median_encoded_Emed,
        handle_panel.get_handle(median_index, -1),
        u_i_list[median_index],
        handle_panel.get_client_i_handles(median_index),
    )
    c_h_L_server = c_h_L.to(Pmed).to(device_panel.server_tee)
    u_i_F_if_valid_list = []
    for i in range(device_panel.client_num):
        accepted, chL_if_accepted = device_panel.server_device(
            package_information, num_returns=2
        )(i, valid_indices, c_h_L_server)
        accepted = accepted.to(device_panel.client_devices[i]).to(
            device_panel.client_tees[i]
        )
        chL_if_accepted = chL_if_accepted.to(device_panel.client_devices[i]).to(
            device_panel.client_tees[i]
        )

        u_i_F_if_valid = device_panel.client_tees[i](clipping)(
            i,
            accepted,
            valid_index_at_clients[i],
            chL_if_accepted,
            u_i_list[i],
            handle_panel.get_handle(i, -1),
            handle_panel.get_handle(i, median_index),
        )
        u_i_F_if_valid_list.append(u_i_F_if_valid)
    return u_i_F_if_valid_list


# Local filtering and aggregation
def local_filtering_and_aggregation(
    device_panel: DevicePanel,
    handle_panel: HandlePanel,
    u_i_F_if_valid_list: List[DeviceObject],
    valid_indices: List[int],
    rng_key: jax.random.PRNGKey,
    params: Params,
    M_old: DeviceObject,
):
    r_dict = {}
    for i in range(device_panel.client_num):
        if valid_indices[i] == 1:
            rng_key, use_rng_key = jax.random.split(rng_key)
            u_i_F_if_valid_list[i] = device_panel.client_tees[i](smallest_noise)(
                u_i_F_if_valid_list[i], params.sigma, use_rng_key
            )
            for j in range(i, device_panel.client_num):
                if valid_indices[j] == 1:
                    rij, rji = corr(
                        params.k,
                        params.m,
                        device_panel.get_tee(i),
                        handle_panel.corr_key_map[i],
                        device_panel.get_tee(j),
                        handle_panel.corr_key_map[j],
                        return_zero_sharing=True,
                    )
                    r_dict[(i, j)] = rij
                    r_dict[(j, i)] = rji

    c_Ut_list = []
    handle_s_h_list = []
    for i in range(device_panel.client_num):
        if valid_indices[i] == 1:
            c_Ut_i = (
                device_panel.get_tee(i)(compute_c_Ut_i)(
                    [r_dict[(i, j)] for j in range(0, i) if valid_indices[j] == 1],
                    [
                        r_dict[(i, j)]
                        for j in range(i + 1, device_panel.client_num)
                        if valid_indices[j] == 1
                    ],
                    u_i_F_if_valid_list[i],
                    handle_panel.get_handle(i, -1),
                )
                .to(device_panel.client_devices[i])
                .to(device_panel.server_device)
                .to(device_panel.server_tee)
            )
            c_Ut_list.append(c_Ut_i)
            handle_s_h_list.append(handle_panel.get_handle(-1, i))
    Mt, ch_Mt_list = aggregation(
        c_Ut_list,
        M_old,
        device_panel.server_tee,
        handle_s_h_list,
        handle_panel.get_server_handles(),
        params,
    )
    return Mt, [
        ch_Mt_list[i]
        .to(device_panel.server_device)
        .to(device_panel.client_devices[i])
        .to(device_panel.client_tees[i])
        for i in range(device_panel.client_num)
    ]


def single_round(
    device_panel: DevicePanel,
    handle_panel: HandlePanel,
    params: Params,
    u_i_list: List[DeviceObject],
    rng: jax.random.PRNGKey,
    M_old: DeviceObject = 0,
    rount_num: int = 0,  # useful for benchmark logging
):
    # Preprocessing
    with time_cost("preprocessing"):
        device_FKeys = key_gen(device_panel, handle_panel, rng)
        abc_pairs = corr_rand_distr_pairwise(device_panel, handle_panel, params)
        sf.wait([device_FKeys, abc_pairs])

    # Local commitment
    with time_cost("local commitment"):
        u_i_list_tee = [
            u_i.to(device_panel.get_tee(i)) for i, u_i in enumerate(u_i_list)
        ]
        sf.wait(u_i_list_tee)

    # Cosine similarity computation with each P𝑗
    with time_cost("cosine similarity computation"):
        cos_sim_pairs = cos_sim_pairwise(
            u_i_list_tee, abc_pairs, device_panel, handle_panel, params
        )
        sf.wait(cos_sim_pairs)

    # Euclidean norm comparison with each P𝑗
    with time_cost("euclidean norm comparison"):
        euclidean_norm_pairs = euclidean_norm_pairwise(
            u_i_list_tee, device_FKeys, device_panel, handle_panel, params
        )
        zij_list = reconstruct_zij_pairwise(
            device_panel, handle_panel, euclidean_norm_pairs, params
        )
        # Compute the final result
        median_index_server_tee = device_panel.server_tee(median_and_index)(
            zij_list, [*range(device_panel.client_num)]
        )
        med_encoded = median_index_encode(
            median_index_server_tee,
            handle_panel.get_server_handles(),
            device_panel.server_tee,
        )
        # everyone will know valid indices anyway
        valid_indices = sf.reveal(
            device_panel.server_tee(server_clustering)(
                device_panel.client_num,
                cos_sim_pairs,
                params.eps,
                params.min_points,
                params.point_num_threshold,
            )
        )

        assert len(valid_indices) == device_panel.client_num, (
            f"{len(valid_indices)}, {device_panel.client_num}"
        )

        valid_index_at_clients = valid_index_encode_devicewise(
            device_panel, handle_panel, valid_indices
        )
        sf.wait(valid_index_at_clients)

    # receive meta info and clipping
    with time_cost("receive meta info and clipping"):
        u_i_F_if_valid_list = receive_meta_info_and_clipping(
            valid_index_at_clients,
            valid_indices,
            med_encoded,
            median_index_server_tee,
            device_panel,
            handle_panel,
            params,
            u_i_list_tee,
        )
        sf.wait(u_i_F_if_valid_list)

    # local filtering and aggregation
    with time_cost("local filtering and aggregation"):
        Mt, c_Mt_list = local_filtering_and_aggregation(
            device_panel,
            handle_panel,
            u_i_F_if_valid_list,
            valid_indices,
            rng,
            params,
            M_old,
        )
        sf.wait(Mt)

    # each client decode Mt_i as output
    with time_cost("output"):
        Mt_list = []
        for i in range(device_panel.client_num):
            Mt_i = device_panel.get_tee(i)(decrypt_to_jnp_array_gcm)(
                c_Mt_list[i], handle_panel.get_handle(i, -1)
            )
            Mt_list.append(Mt_i)
        sf.wait(Mt_list)
    return Mt, Mt_list


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


def main():
    device_panel, handle_pannel, params = sf_setup(edge_parties_number=20, m=500000)
    rng_key = jax.random.PRNGKey(0)
    Mt = 0
    for i in range(1):
        # do training and get u_i_list
        u_i_list = simulate_data_u(device_panel, -1.0, 1.0, params.m)
        # do clustering and get Mt
        with time_cost(f"single round {i}"):
            Mt, Mt_list = single_round(
                device_panel, handle_pannel, params, u_i_list, rng_key, Mt, i
            )


def main_prod(sf_config: dict, self_party: str, m: int):
    device_panel, handle_pannel, params = sf_setup_prod(sf_config, self_party, m)
    rng_key = jax.random.key(0)
    Mt = 0
    for i in range(1):
        # do training and get u_i_list
        u_i_list = simulate_data_u(device_panel, -1.0, 1.0, params.m)
        # do clustering and get Mt
        with time_cost(f"single round {i}"):
            Mt, Mt_list = single_round(
                device_panel, handle_pannel, params, u_i_list, rng_key, Mt, i
            )
    sf.shutdown()


if __name__ == "__main__":
    main()
