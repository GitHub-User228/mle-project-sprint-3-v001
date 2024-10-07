"""
This script can be used to use test the fastapi app by sending requests
from multiple IP addresses or from a single IP address. Each request has
different model_params - it is drawn from a corresponding generator.

Args:
    --r (int): 
        Number of requests to send. Must be a positive integer 
        and less than MAX_REQUESTS. Default is 10.
    --d (float):
        Delay between requests in seconds. Must be a non-negative float
        and less than MAX_DELAY. Default is 0.5.
    --m (bool):
        Whether to use multiple IP addresses. Default is False.
    --i (float):
        Rate of invalid requests. Must be non-negative float and less than 1.
        Default is 0.
"""

import time
import random
import argparse
import requests

from scripts.env import env_vars
from scripts.settings import config, generate_random_model_params
from scripts.utils import (
    boolean,
    generate_random_ip,
)


# Constants for default values and limits
DEFAULT_REQUESTS = 10
DEFAULT_DELAY = 0.5
DEFAULT_MULTIPLE_IPS = False
DEFAULT_INVALID_RATE = 0.0
MAX_REQUESTS = 50
MAX_DELAY = 10


def main():

    # Adding argument for to define number of requests
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("-d", "--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument(
        "-m", "--multiple-ips", type=boolean, default=DEFAULT_MULTIPLE_IPS
    )
    parser.add_argument(
        "-i", "--invalid-rate", type=float, default=DEFAULT_INVALID_RATE
    )

    # Specifying the URL and headers for the requests
    url = (
        f"http://localhost:{env_vars.app_vm_port}"
        f"{config['prediction_endpoint']}"
    )
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-Forwarded-For": generate_random_ip(),
    }

    # Parsing and validating the argument if specified
    args = parser.parse_args()
    if not 1 <= args.requests <= MAX_REQUESTS:
        raise ValueError(
            f"Number of requests must be between 1 and {MAX_REQUESTS}."
        )
    if not 0 <= args.delay <= MAX_DELAY:
        raise ValueError(f"Delay must be between 0 and {MAX_DELAY}.")
    if not 0 <= args.invalid_rate <= 1:
        raise ValueError(f"Invalid rate must be between 0 and 1.")

    # Sending requests
    for _ in range(args.requests):
        if args.multiple_ips:
            headers["X-Forwarded-For"] = generate_random_ip()
        response = requests.post(
            url=f"{url}/?user_id={random.randint(1, 100000)}",
            json=generate_random_model_params(invalid_rate=args.invalid_rate),
            headers=headers,
        )
        print(
            f"IP: {headers['X-Forwarded-For']}; "
            f"Status code: {response.status_code}; "
            f"Response body: {response.json()}"
        )
        time.sleep(args.delay)


if __name__ == "__main__":
    main()
