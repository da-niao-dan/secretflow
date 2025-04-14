import socket
from typing import List, Tuple

import jax
import jax.numpy as jnp
from pydantic import BaseModel
import spu
from utils import Devices, Handles, Params, get_random_bytes

import secretflow as sf
from secretflow import Device


class DevicePanel:
    def __init__(
        self,
        client_devices: List[Device],
        server_device: Device,
        client_tees: List[Device],
        server_tee: Device,
    ):
        assert len(client_tees) == len(client_devices)
        self.client_devices = client_devices
        self.server_device = server_device
        self.client_tees = client_tees
        self.server_tee = server_tee
        self.client_num = len(client_tees)

    def enumerate_pairs(self):
        # Iterate over diagonals, starting from 1 to client_num-1
        for k in range(1, self.client_num):
            for i in range(self.client_num):
                j = (i + k) % self.client_num
                if i < j:  # Ensure each pair (i, j) is computed only once
                    yield (i, j)

    def get_device(self, i):
        if i == -1:
            return self.server_device
        return self.client_devices[i]

    def get_tee(self, i):
        if i == -1:
            return self.server_tee
        return self.client_tees[i]

    def get_device_pair(self, i, j):
        return (self.get_device(i), self.get_device(j))

    def get_tee_pair(self, i, j):
        return (self.get_tee(i), self.get_tee(j))

    def build_devices(self, i, j) -> Devices:
        return Devices(
            self.get_device(i),
            self.get_device(j),
            self.server_device,
            self.get_tee(i),
            self.get_tee(j),
            self.server_tee,
        )

    def enumerate_device_pairs(self):
        for i, j in self.enumerate_pairs():
            yield (self.build_devices(i, j), self.build_handles(i, j))

    def enumerate_tee_pairs(self):
        for i, j in self.enumerate_pairs():
            yield (self.client_tees[i], self.client_tees[j])


class HandlePanel:
    def __init__(self, device_panel: DevicePanel, kappa: int):
        self.handle_map = {}
        self.corr_key_map = {}
        self.client_num = device_panel.client_num
        for (i, j), (tee_i, tee_j) in zip(
            device_panel.enumerate_pairs(), device_panel.enumerate_tee_pairs()
        ):
            handle_i_j = tee_i(lambda x: get_random_bytes(x))(kappa)
            self.handle_map.update({(i, j): handle_i_j, (j, i): handle_i_j.to(tee_j)})

        for i, client_tee in enumerate(device_panel.client_tees):
            handle_i_s = client_tee(lambda x: get_random_bytes(x))(kappa)
            self.handle_map.update(
                {(i, -1): handle_i_s, (-1, i): handle_i_s.to(device_panel.server_tee)}
            )

        # for some  strange encoding sent to oneself
        for i in range(self.client_num):
            self.handle_map.update(
                {
                    (i, i): device_panel.client_tees[i](lambda x: get_random_bytes(x))(
                        kappa
                    )
                }
            )

        for i in range(self.client_num):
            self.corr_key_map.update(
                {i: device_panel.client_tees[i](lambda x: jax.random.key(x))(kappa)}
            )
        self.corr_key_map[-1] = device_panel.server_tee(lambda x: jax.random.key(x))(
            kappa
        )

    def get_handle(self, i, j):
        return self.handle_map[(i, j)]

    def get_server_handles(self):
        return [self.get_handle(-1, i) for i in range(self.client_num)]

    def get_client_i_handles(self, i):
        return [self.get_handle(i, j) for j in range(self.client_num)]

    def build_handles(self, i, j) -> Handles:
        return Handles(
            self.get_handle(-1, i),
            self.get_handle(-1, j),
            self.get_handle(i, -1),
            self.get_handle(j, -1),
            self.get_handle(j, i),
            self.get_handle(i, j),
            self.corr_key_map[i],
            self.corr_key_map[j],
            self.corr_key_map[-1],
        )


def sf_setup(
    edge_parties_number=2,
    edge_party_name="edge_party_{i}",
    server_party_name="server_party",
    fxp=26,
    fxp_type=jnp.uint64,
    kappa=32,
    k=64,
    m=10,
):
    edge_parties = [edge_party_name.format(i=i) for i in range(edge_parties_number)]
    server_party = [server_party_name]
    all_parties = edge_parties + server_party
    sf.init(
        parties=all_parties,
        address="local",
        omp_num_threads=len(all_parties),
        cross_silo_comm_backend="brpc_link",
        ray_mode=True,
        enable_waiting_for_other_parties_ready=False,
        cross_silo_comm_options={
            "proxy_max_restarts": 3,
            "timeout_in_ms": 300 * 1000,
            # Give recv_timeout_ms a big value, e.g.
            # The server does nothing but waits for task finish.
            # To fix the psi timeout, got a week here.
            "recv_timeout_ms": 7 * 24 * 3600 * 1000,
            "connect_retry_times": 3600,
            "connect_retry_interval_ms": 1000,
            "brpc_channel_protocol": "http",
            "brpc_channel_connection_type": "pooled",
        },
    )
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
        fxp=fxp,
        fxp_type=fxp_type,
        kappa=kappa,
        k=k,
        m=m,
        eps=10e-5,
        min_points=max(int(edge_parties_number / 4), 1),
        point_num_threshold=max(int(edge_parties_number / 2), 1),
    )

    device_panel = DevicePanel(edge_devices, server_device, edge_tees, server_tee)
    handle_panel = HandlePanel(device_panel, kappa)
    return device_panel, handle_panel, params


def get_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class PartyNames(BaseModel):
    client_names: List[str]
    server_names: List[str]

    def __iter__(self):
        return iter(self.client_names + self.server_names)


def create_party_names(client_num: int, server_num: int = 1) -> PartyNames:
    return PartyNames(
        client_names=[f"client_{i}" for i in range(client_num)], server_names=[f"server_{i}" for i in range(server_num)]
    )


def create_static_config(party_names: PartyNames):
    party_ports = {party: get_available_port() for party in party_names}
    spu_ports = {party: get_available_port() for party in party_names.server_names}

    config = {}

    config["parties"] = {
        party_name: {
            "address": f"127.0.0.1:{party_ports[party_name]}",
            "listen_addr": f"0.0.0.0:{party_ports[party_name]}",
        }
        for party_name in party_names
    }

    config["nodes"] = [
        {"party": party_name, "address": f"127.0.0.1:{spu_ports[party_name]}"}
        for party_name in party_names.server_names
    ]

    return config


def sf_setup_prod(
    sf_config: dict,
    party_name: str,
    m: int = 10,
):
    party_names = create_party_names(len(sf_config["parties"]) - 1)
    sf_config["self_party"] = party_name
    sf.init(
        cluster_config=sf_config,
        cross_silo_comm_backend="brpc_link",
        ray_mode=False,
        cross_silo_comm_options={
            "proxy_max_restarts": 3,
            "timeout_in_ms": 30 * 1000,
            # Give recv_timeout_ms a big value, e.g.
            # The server does nothing but waits for task finish.
            # To fix the psi timeout, got a week here.
            "recv_timeout_ms": 7 * 24 * 3600 * 1000,
            "connect_retry_times": 360,
            "connect_retry_interval_ms": 100,
            "brpc_channel_protocol": "http",
            "brpc_channel_connection_type": "pooled",
            "exit_on_sending_failure": True,
            "http_max_payload_size": 5 * 1024 * 1024,
        },
        enable_waiting_for_other_parties_ready=True,
    )
    edge_devices = [
        sf.PYU(edge_party_name) for edge_party_name in party_names.client_names
    ]
    server_device = sf.PYU(party_names.server_names[0])
    # use pyu to simulate teeu
    edge_tees = [
        sf.PYU(edge_party_name) for edge_party_name in party_names.client_names
    ]

    server_device = sf.PYU(party_names.server_names[0])
    server_tee = sf.PYU(party_names.server_names[0])
    kappa = 32
    params = Params(
        fxp=26,
        fxp_type=jnp.uint64,
        kappa=kappa,
        k=64,
        m=m,
        eps=10e-5,
        min_points=max(int(len(party_names.client_names) / 4), 1),
        point_num_threshold=max(int(len(party_names.client_names) / 2), 1),
    )

    device_panel = DevicePanel(edge_devices, server_device, edge_tees, server_tee)
    handle_panel = HandlePanel(device_panel, kappa)
    return device_panel, handle_panel, params


def sf_setup_mpc(
    sf_config: dict,
    party_names: PartyNames,
    n: int = 2,
    m: int = 10,
):
    sf.init(
        cluster_config=sf_config,
        cross_silo_comm_backend="brpc_link",
        ray_mode=False,
        cross_silo_comm_options={
            "proxy_max_restarts": 3,
            "timeout_in_ms": 30 * 1000,
            "recv_timeout_ms": 7 * 24 * 3600 * 1000,
            "connect_retry_times": 360,
            "connect_retry_interval_ms": 100,
            "brpc_channel_protocol": "http",
            "brpc_channel_connection_type": "pooled",
            "exit_on_sending_failure": True,
            "http_max_payload_size": 5 * 1024 * 1024,
        },
        enable_waiting_for_other_parties_ready=True,
    )

    kappa = 32
    params = Params(
        fxp=26,
        fxp_type=jnp.uint64,
        kappa=kappa,
        k=64,
        m=m,
        eps=10e-5,
        min_points=max(int(n / 4), 1),
        point_num_threshold=max(int(n / 2), 1),
    )

    pyu_devices = [sf.PYU(server) for server in party_names.server_names]
    spu_config = sf.utils.testing.cluster_def(party_names.server_names)
    spu_config['nodes'] = sf_config['nodes']
    spu_config['runtime_config']['field'] = spu.spu_pb2.FM64
    spu_config['runtime_config']['fxp_fraction_bits'] = params.fxp
    server_spu = sf.SPU(spu_config)

    device_panel = DevicePanel(pyu_devices, server_spu, pyu_devices, server_spu)
    return device_panel, params