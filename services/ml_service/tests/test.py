"""
This script can be used to use test the fastapi app by sending requests
from multiple IP addresses or from a single IP address. Each request has
different model_params - it is drawn from a corresponding generator.

Args:
    --r (int): 
        Number of requests to send. Must be a positive integer 
        and less than 10. Default is 10.
    --d (float):
        Delay between requests in seconds. Must be a non-negative float
        and less than 50. Default is 0.5.
    --m (bool):
        Whether to use multiple IP addresses. Default is False.
    --i (float):
        Rate of invalid requests. Must be non-negative float and less than 1.
        Default is 0.
"""

import sys
import time
import random
import argparse
import requests
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.env import env_vars
from scripts.utils import boolean
from scripts.settings import config
from scripts.generators import generate_random_ip, generate_random_model_params

# Adding argument for to define number of requests
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--requests", type=int)
parser.add_argument("-d", "--delay", type=float)
parser.add_argument("-m", "--multiple-ips", type=boolean)
parser.add_argument("-i", "--invalid-rate", type=float)

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
if args.requests is not None:
    if args.requests < 1 or args.requests > 50:
        raise ValueError("Number of requests must be between 1 and 50.")
    n_requests = args.requests
else:
    n_requests = 10
if args.delay is not None:
    if args.delay < 0 or args.delay > 10:
        raise ValueError("Delay must be between 0 and 10.")
    delay = args.delay
else:
    delay = 0.5
if args.multiple_ips is not None:
    use_multiple_ips = args.multiple_ips
else:
    use_multiple_ips = False
if args.invalid_rate is not None:
    invalid_rate = args.invalid_rate
else:
    invalid_rate = 0.0
if not 0 <= invalid_rate <= 1:
    raise ValueError("Invalid rate must be between 0 and 1.")


# Sending requests
for _ in range(n_requests):
    if use_multiple_ips:
        headers["X-Forwarded-For"] = generate_random_ip()
    response = requests.post(
        url=f"{url}/?user_id={random.randint(1, 100000)}",
        json=generate_random_model_params(invalid_rate=invalid_rate),
        headers=headers,
    )
    print(
        f"IP: {headers['X-Forwarded-For']}; "
        f"Status code: {response.status_code}; "
        f"Response body: {response.json()}"
    )
    time.sleep(delay)
