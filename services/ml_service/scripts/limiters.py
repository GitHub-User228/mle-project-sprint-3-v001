from fastapi import Request
from fastapi_limiter.depends import RateLimiter

from scripts.settings import (
    get_global_request_rate_limit_settings,
    get_ip_request_rate_limit_settings,
)
from scripts.callbacks import global_callback, per_ip_callback


async def get_ip_key(request: Request) -> str:
    """
    Extracts the IP address from the incoming request, handling cases
    where the IP address is forwarded.

    Args:
        request (Request):
            The FastAPI request object containing
            information about the incoming request.

    Returns:
        str:
            The client's IP address as a string.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host + ":" + request.scope["path"]


async def get_default_ip_key(request: Request) -> str:
    """
    Generates a default IP key for the global rate limiter.

    Args:
        request (Request):
            The FastAPI request object containing information about
            the incoming request.

    Returns:
        str:
            A default IP key value of "default".
    """
    return "default"


# Rate limiter for global requests
global_limiter = RateLimiter(
    **get_global_request_rate_limit_settings().dict(),
    identifier=get_default_ip_key,
    callback=global_callback,
)

# Rate limiter for requests per IP address
ip_limiter = RateLimiter(
    **get_ip_request_rate_limit_settings().dict(),
    identifier=get_ip_key,
    callback=per_ip_callback,
)
