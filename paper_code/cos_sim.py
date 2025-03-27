from typing import Tuple

import jax
import jax.numpy as jnp
from utils import (
    Devices,
    Handles,
    Params,
    corr,
    corr_rand_distribute,
    decrypt_to_jnp_array_gcm,
    encrypt_jnp_array_gcm,
    gen_handles,
    simulate_data_u,
)

import secretflow as sf
from secretflow.device import DeviceObject

jax.config.update("jax_enable_x64", True)


def cos_sim(
    u_i,
    u_j,
    devices: Devices,
    handles: Handles,
    params: Params,
    abc_info: Tuple[DeviceObject],
    verbose=False,
) -> DeviceObject:
    # programming details not related to protocol
    if verbose:
        print("input: ", sf.reveal(u_i), sf.reveal(u_j))

    params.fxp_type = jnp.uint64

    server_a, server_b, c, edge_tee_i_a, edge_tee_j_b = abc_info

    if verbose:
        print()
        print("preprocessing: ")
        print("server_a: ", sf.reveal(server_a))
        print("server_b: ", sf.reveal(server_b))

        print("edge_tee_i_a: ", sf.reveal(server_a))
        print("edge_tee_j_b: ", sf.reveal(edge_tee_j_b))
        print("c: ", sf.reveal(c))

    # normalize u_i and u_j
    u_i_normalized = devices.edge_tee_i(
        lambda x: jnp.array(
            x / jnp.linalg.norm(x) * (2.0**params.fxp), dtype=params.fxp_type
        )
    )(u_i.to(devices.edge_tee_i))
    u_j_normalized = devices.edge_tee_j(
        lambda x: jnp.array(
            x / jnp.linalg.norm(x) * (2.0**params.fxp), dtype=params.fxp_type
        )
    )(u_j.to(devices.edge_tee_j))

    if verbose:
        print()
        print("step 1:")
        print("params.fxp", params.fxp)
        print("u_i_normalized", sf.reveal(u_i_normalized))
        print("u_j_normalized", sf.reveal(u_j_normalized))

    # E_i encrypts e = u_i_normalized - a, sends to P_j via P_i
    e = devices.edge_tee_i(lambda x, y: x - y)(u_i_normalized, edge_tee_i_a)

    if verbose:
        print()
        print("step 2:")
        print("a: ", sf.reveal(edge_tee_i_a))
        print("e = u_i_normalized - a: ", sf.reveal(e))

    c_e = devices.edge_tee_i(encrypt_jnp_array_gcm)(e, handles.i_j)
    c_e_j = c_e.to(devices.edge_device_i).to(devices.edge_device_j)

    # E_j encrypts f = u_j_normalized - b, sends to P_i via P_j
    f = devices.edge_tee_j(lambda x, y: x - y)(u_j_normalized, edge_tee_j_b)
    if verbose:
        print("step 2, f:")
        print("f dtype", sf.reveal(f).dtype)
    c_f = devices.edge_tee_j(encrypt_jnp_array_gcm)(f, handles.j_i)

    if verbose:
        print()
        print("step 3:")
        print("u_j_normalized: ", sf.reveal(u_j_normalized))
        print("b: ", sf.reveal(edge_tee_j_b))
        print("f = u_j_normalized - b: ", sf.reveal(f))

    c_f_i = c_f.to(devices.edge_device_j).to(devices.edge_device_i)

    f_dec = devices.edge_tee_i(decrypt_to_jnp_array_gcm)(
        c_f_i.to(devices.edge_tee_i), handles.i_j
    )

    e_dec = devices.edge_tee_j(decrypt_to_jnp_array_gcm)(
        c_e_j.to(devices.edge_tee_j),
        handles.j_i,
    )

    edge_tee_i_a_1, edge_tee_j_a_1 = corr(
        params.k,
        params.m,
        devices.edge_tee_i,
        handles.i,
        devices.edge_tee_j,
        handles.j,
    )
    edge_tee_i_b_0, edge_tee_j_b_0 = corr(
        params.k,
        params.m,
        devices.edge_tee_i,
        handles.i,
        devices.edge_tee_j,
        handles.j,
    )
    edge_tee_i_d, edge_tee_j_d = corr(
        params.k,
        params.m,
        devices.edge_tee_i,
        handles.i,
        devices.edge_tee_j,
        handles.j,
        True,
    )

    edge_tee_i_a_0 = devices.edge_tee_i(lambda x, y: params.fxp_type(x - y))(
        edge_tee_i_a, edge_tee_i_a_1
    )
    edge_tee_j_b_1 = devices.edge_tee_j(lambda x, y: params.fxp_type(x - y))(
        edge_tee_j_b, edge_tee_j_b_0
    )

    if verbose:
        print()
        print("step 6:")
        print("a_0: ", sf.reveal(edge_tee_i_a_0))
        print("b_1: ", sf.reveal(edge_tee_j_b_1))
        print("a_1: ", sf.reveal(edge_tee_i_a_1))
        print("b_0: ", sf.reveal(edge_tee_j_b_0))
        print("d_0: ", sf.reveal(edge_tee_i_d))
        print("d_1: ", sf.reveal(edge_tee_j_d))

    # E_i computes:
    z_bracket_0 = devices.edge_tee_i(
        lambda x1, x2, x3, x4, x5: params.fxp_type(x1 * x2)
        + params.fxp_type(x3 * x4)
        + params.fxp_type(x1 * x3)
        + params.fxp_type(x5)
    )(e, edge_tee_i_b_0, f_dec, edge_tee_i_a_0, edge_tee_i_d)

    if verbose:
        print()
        print("step 7:")
        print("e", sf.reveal(e))
        print("f_dec", sf.reveal(f_dec))
        print("edge_tee_i_d", sf.reveal(edge_tee_i_d))
        print("z_bracket_0", sf.reveal(z_bracket_0))

    # E_i encrypts z_bracket_0, sends to server tee via server
    c_z_bracket_0 = devices.edge_tee_i(encrypt_jnp_array_gcm)(z_bracket_0, handles.i_s)
    c_z_bracket_0_server = c_z_bracket_0.to(devices.server_device).to(
        devices.server_tee
    )

    # E_j computes:
    z_bracket_1 = devices.edge_tee_j(
        lambda x1, x2, x3, x4, x5: params.fxp_type(
            params.fxp_type(x1 * x2) + params.fxp_type(x3 * x4) + params.fxp_type(x5)
        )
    )(e_dec, edge_tee_j_b_1, f, edge_tee_j_a_1, edge_tee_j_d)

    # E_j encrypts z_bracket_1, sends to server tee via server
    c_z_bracket_1 = devices.edge_tee_j(
        encrypt_jnp_array_gcm,
    )(z_bracket_1, handles.j_s)
    c_z_bracket_1_server = c_z_bracket_1.to(devices.server_device).to(
        devices.server_tee
    )

    if verbose:
        print()
        print("step 8:")
        print("e_dec", sf.reveal(e_dec))
        print("f", sf.reveal(f))
        print("edge_tee_j_d", sf.reveal(edge_tee_j_d))
        print("z_bracket_1", sf.reveal(z_bracket_1))

    # server tries to decrypt
    z_bracket_0_dec = devices.server_tee(decrypt_to_jnp_array_gcm)(
        c_z_bracket_0_server,
        handles.s_i,
    )
    z_bracket_1_dec = devices.server_tee(decrypt_to_jnp_array_gcm)(
        c_z_bracket_1_server,
        handles.s_j,
    )

    z = devices.server_tee(lambda x, y: params.fxp_type(x + y))(
        z_bracket_0_dec, z_bracket_1_dec
    )
    cos_sim_val = devices.server_tee(lambda x, y: jnp.sum(params.fxp_type(x + y)))(z, c)

    if verbose:
        print()
        print("step 10:")
        print("z_bracket_0_dec", sf.reveal(z_bracket_0_dec))
        print("z_bracket_1_dec", sf.reveal(z_bracket_1_dec))
        print("z", sf.reveal(z))
        print("c", sf.reveal(c))
        print("cos_sim_val", sf.reveal(cos_sim_val))
        print("params.fxp", params.fxp)
        print(
            "cos_sim_val / 2^(2*params.fxp)",
            sf.reveal(cos_sim_val) / (2.0 ** (2 * params.fxp)),
        )
    return cos_sim_val


def main():
    from performance_stats import time_cost

    import secretflow as sf

    edge_parties_number = 2
    edge_party_name = "edge_party_{i}"
    edge_parties = [edge_party_name.format(i=i) for i in range(edge_parties_number)]
    server_party_name = "server_party"
    server_party = [server_party_name]
    all_parties = edge_parties + server_party
    sf.init(parties=all_parties, address="local")
    edge_devices = [
        sf.PYU(edge_party_name.format(i=i)) for i in range(edge_parties_number)
    ]

    # use pyu to simulate teeu
    edge_tees = [
        sf.PYU(edge_party_name.format(i=i)) for i in range(edge_parties_number)
    ]

    server_device = sf.PYU(server_party_name)
    server_tee = sf.PYU(server_party_name)
    params = Params(
        fxp=26,
        fxp_type=jnp.uint64,
        kappa=32,
        k=64,
        m=500000,
        eps=10e-5,
        min_points=3,
        point_num_threshold=3,
    )

    # custom parameters
    i = 0
    j = 1
    u_low = 0
    u_high = 1

    devices = Devices(
        edge_devices[i],
        edge_devices[j],
        server_device,
        edge_tees[i],
        edge_tees[j],
        server_tee,
    )
    handles = gen_handles(devices, params)
    u_i, u_j = simulate_data_u(devices, u_low, u_high, params.m)
    abc_info = corr_rand_distribute(devices, handles, params)
    cos_sim_1 = cos_sim(u_i, u_j, devices, handles, params, abc_info, verbose=True)
    cos_sim_2 = cos_sim(u_i, u_j, devices, handles, params, abc_info, verbose=False)
    with time_cost("cos_sim"):
        print(sf.reveal([cos_sim_1, cos_sim_2]))


if __name__ == "__main__":
    main()
