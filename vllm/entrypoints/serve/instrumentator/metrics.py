# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import regex as re
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from starlette.routing import Mount

from vllm.v1.metrics.prometheus import get_prometheus_registry


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""

    registry = get_prometheus_registry()

    # Add prometheus asgi route without the request-instrumentation middleware.
    # prometheus-fastapi-instrumentator 8.0.0 assumes every Starlette route has
    # a `.path` attribute, which is no longer true for FastAPI included routers.
    # That middleware can 500 every API request before it reaches vLLM.
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
