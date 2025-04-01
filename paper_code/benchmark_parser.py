import argparse
import logging
import re

from benchmark_result import BenchmarkResults


def parse_log(log_file, n_clients, n_inputs):
    client_data = {
        "Time_Preprocessing": 0,
        "Sent_Preprocessing": 0,
        "Time_Local_Commitment": 0,
        "Sent_Local_Commitment": 0,
        "Time_Cosine_Similarity": 0,
        "Sent_Cosine_Similarity": 0,
        "Time_Euclidean_Norm": 0,
        "Sent_Euclidean_Norm": 0,
        "Time_Meta_Clipping": 0,
        "Sent_Meta_Clipping": 0,
        "Time_Aggregation": 0,
        "Sent_Aggregation": 0,
        "Time_Total": 0,
        "Sent_Total": 0,
    }

    phase_mapping = {
        "preprocessing": "Preprocessing",
        "local commitment": "Local_Commitment",
        "cosine similarity computation": "Cosine_Similarity",
        "euclidean norm comparison": "Euclidean_Norm",
        "receive meta info and clipping": "Meta_Clipping",
        "local filtering and aggregation": "Aggregation",
    }

    current_phase = None
    total_bytes_transferred = 0

    with open(log_file, "r") as f:
        for line in f:
            # Extract time costs for different phases (using client_0 as representative)
            if "[client_0]" in line:
                time_match = re.search(
                    r"-- Time cost for '(.*?)': (\d+\.\d+) seconds", line
                )
                if time_match:
                    phase = time_match.group(1)
                    time_cost = float(time_match.group(2))
                    if phase in phase_mapping:
                        client_data[f"Time_{phase_mapping[phase]}"] = time_cost
                    current_phase = phase_mapping.get(phase, None)

            # Extract bytes transferred for all parties
            bytes_match = re.search(r"bytes transferred: (\d+)", line)
            if bytes_match:
                bytes_transferred = int(bytes_match.group(1))
                total_bytes_transferred += bytes_transferred

            # Attribute the total bytes transferred to the current phase
            if current_phase and "Finished reveal bytes transferred:" in line:
                sent_key = f"Sent_{current_phase}"
                if sent_key in client_data:
                    client_data[sent_key] += total_bytes_transferred
                total_bytes_transferred = 0  # Reset for the next phase

    client_data["N_Clients"] = n_clients
    client_data["N_Inputs"] = n_inputs

    # Calculate total time and sent bytes
    client_data["Time_Total"] = sum(
        client_data[key]
        for key in client_data
        if key.startswith("Time_") and key != "Time_Total"
    )
    client_data["Sent_Total"] = sum(
        client_data[key]
        for key in client_data
        if key.startswith("Sent_") and key != "Sent_Total"
    )

    return client_data


def main():
    parser = argparse.ArgumentParser(description="Process benchmark log files.")
    parser.add_argument(
        "--log_file", type=str, required=True, help="Path to the log file."
    )
    parser.add_argument(
        "--n_clients", type=int, required=True, help="Number of clients."
    )
    parser.add_argument("--n_inputs", type=int, required=True, help="Number of inputs.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        # Parse log and create the model
        benchmark_results_data = parse_log(args.log_file, args.n_clients, args.n_inputs)
        benchmark_results = BenchmarkResults(**benchmark_results_data)

        # Print the result
        print("Parsed Results:")
        print(benchmark_results)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
