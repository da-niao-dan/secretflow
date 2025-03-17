from typing import List, Tuple

import jax.numpy as jnp
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
        for i in range(self.client_num):
            for j in range(i + 1, self.client_num):
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
        for i in range(self.client_num):
            for j in range(i + 1, self.client_num):
                yield (self.client_devices[i], self.client_devices[j])

    def enumerate_tee_pairs(self):
        for i in range(self.client_num):
            for j in range(i + 1, self.client_num):
                yield (self.client_tees[i], self.client_tees[j])


class HandlePanel:
    def __init__(self, device_panel: DevicePanel, kappa: int):
        self.handle_map = {}
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
        print("Handle map", self.handle_map)

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
        )


def sf_setup(
    edge_parties_number=2,
    edge_party_name='edge_party_{i}',
    server_party_name='server_party',
    fxp=26,
    fxp_type=jnp.uint64,
    kappa=32,
    k=64,
    m=10,
):
    edge_parties = [edge_party_name.format(i=i) for i in range(edge_parties_number)]
    server_party = [server_party_name]
    all_parties = edge_parties + server_party
    sf.init(parties=all_parties, address='local')
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
