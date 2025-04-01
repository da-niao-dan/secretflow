# this script will create sf cluster config for n parties
# then use python script to create multiple threads with corresponding config files
from dotenv import load_dotenv

load_dotenv()

import argparse
from multiprocessing import Manager, Process, set_start_method

from device_setups import create_party_names, create_static_config
from FL_program import main_prod

# jax is multithreaded, we need to avoid os.fork()
set_start_method("spawn", force=True)


def party_execute(config, party_name, m):
    config["self_party"] = party_name
    main_prod(config, party_name, m)


def each_party_run(n: int, m: int):
    party_names = create_party_names(n)
    config = create_static_config(party_names)

    processes = []
    for party in party_names:
        process = Process(target=party_execute, args=(config, party, m))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="number of parties")
    parser.add_argument("--m", type=int, help="vector size of gradient")
    args = parser.parse_args()

    each_party_run(args.n, args.m)
