import logging
import time
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


@contextmanager
def time_cost(section_name):
    logging.info(f"Starting '{section_name}'...")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        # Use logging instead of print
        logging.info(f"Time cost for '{section_name}': {duration:.6f} seconds")
    logging.info(f"Finished '{section_name}'.")


# Example usage
if __name__ == "__main__":
    with time_cost("example section"):
        # Place your code here
        total = 0
        for i in range(1000000):
            total += i
