from fastapi import Request, Response, HTTPException

from scripts import logger
from scripts.utils import calculate_expire_time
from scripts.prometheus_metrics import TOO_MANY_REQUESTS_COUNT


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
