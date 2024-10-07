from fastapi_limiter.depends import RateLimiter
from fastapi import Request, Response, HTTPException

from scripts import logger
from scripts.utils import calculate_expire_time
from scripts.metrics import TOO_MANY_REQUESTS_COUNT
from scripts.settings import (
    get_global_request_rate_limit_settings,
    get_ip_request_rate_limit_settings,
)


async def global_callback(
    request: Request, response: Response, pexpire: int
) -> None:
    """
    Logs a warning message and raises an HTTP 429 Too Many Requests
    exception with the provided message and a Retry-After header set
    to the specified expire time.

    Args:
        request (Request):
            The incoming request object.
        response (Response):
            The outgoing response object.
        pexpire (int):
            The number of seconds the client should wait before retrying
            the request.
    """
    expire = calculate_expire_time(pexpire)
    message = f"Too Many Overall Requests. Retry after {expire} seconds."
    logger.warning(message)
    TOO_MANY_REQUESTS_COUNT.inc()
    raise HTTPException(
        status_code=429,
        detail=message,
        headers={"Retry-After": str(expire)},
    )


async def per_ip_callback(
    request: Request, response: Response, pexpire: int
) -> None:
    """
    Logs a warning message and raises an HTTP 429 Too Many Requests
    exception with the provided message and a Retry-After header set
    to the specified expire time for requests from a specific IP address.

    Args:
        request (Request):
            The incoming request object.
        response (Response):
            The outgoing response object.
        pexpire (int):
            The number of seconds the client should wait before retrying
            the request.
    """
    expire = calculate_expire_time(pexpire)
    message = (
        f"Too Many Requests from your IP. " f"Retry after {expire} seconds."
    )
    logger.warning(message)
    TOO_MANY_REQUESTS_COUNT.inc()
    raise HTTPException(
        status_code=429,
        detail=message,
        headers={"Retry-After": str(expire)},
    )


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
