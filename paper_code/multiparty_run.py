# this script will create sf cluster config for n parties
# then use python script to create multiple threads with corresponding config files
from dotenv import load_dotenv

load_dotenv()

import argparse
from multiprocessing import Manager, Process, set_start_method

from device_setups import create_party_names, create_static_config
from FL_program import main_prod as main_prod_slime
from baseline_tee import main_prod as main_prod_tee

# jax is multithreaded, we need to avoid os.fork()
set_start_method("spawn", force=True)


def party_execute(config, party_name, m, f):
    config["self_party"] = party_name
    if f == "slime":
        main_prod_slime(config, party_name, m)
    elif f == "tee":
        main_prod_tee(config, party_name, m)
    else:
        raise ValueError(f"Unknown testing framework: {f}")


def each_party_run(n: int, m: int, f: str):
    party_names = create_party_names(n)
    config = create_static_config(party_names)

    processes = []
    for party in party_names:
        process = Process(target=party_execute, args=(config, party, m, f))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="number of parties")
    parser.add_argument("--m", type=int, help="vector size of gradient")
    parser.add_argument("--f", type=str, default="slime", help="testing framework: ['slime', 'tee']")
    args = parser.parse_args()

    each_party_run(args.n, args.m, args.f)
