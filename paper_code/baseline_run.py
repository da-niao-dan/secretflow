# this script will create sf cluster config for n parties
# then use python script to create multiple threads with corresponding config files
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

import argparse
from multiprocessing import Manager, Process, set_start_method

from device_setups import get_available_port
from baseline_tee import main_prod
from baseline_mpc import main_mpc

# jax is multithreaded, we need to avoid os.fork()
set_start_method("spawn", force=True)


class PartyNames(BaseModel):
    client_names: List[str]
    server_names: List[str]

    def __iter__(self):
        return iter(self.client_names + self.server_names)


def create_party_names(client_num: int, server_num: int=1) -> PartyNames:
    return PartyNames(
        client_names=[f"client_{i}" for i in range(client_num)], server_names=[f"server_{i}" for i in range(server_num)]
    )


def create_static_config(party_names):
    party_ports = {party: get_available_port() for party in party_names}
    spu_ports = {party: get_available_port() for party in party_names}

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
        for party_name in party_names
    ]

    return config


def party_execute(config, party_names, self_party, n, m):
    config["self_party"] = self_party
    main_mpc(config, party_names, n, m)


def each_party_run(n: int, m: int, s: int=1):
    party_names = create_party_names(n, s)
    config = create_static_config(party_names)

    processes = []
    for party in party_names:
        process = Process(target=party_execute, args=(config, party_names, party, m))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

def each_server_run(n: int, m: int, s: int=3):
    party_names = create_party_names(n, s)
    config = create_static_config(party_names.server_names)

    processes = []
    for server in party_names.server_names:
        process = Process(target=party_execute, args=(config, party_names.server_names, server, n, m))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, help="number of servers", default=3)
    parser.add_argument("--n", type=int, help="number of parties")
    parser.add_argument("--m", type=int, help="vector size of gradient")
    args = parser.parse_args()

    each_server_run(args.n, args.m, args.s)
