#!/bin/bash
set -e

log() {
    echo "$(date): $1" >&2
}

# Check if required environment variables are set
if [ -z "${APP_DOCKER_PORT}" ]; then
    log "Error: APP_DOCKER_PORT is not set"
    exit 1
fi

if [ -z "${LOG_DIR}" ]; then
    log "Error: LOG_DIR is not set"
    exit 1
fi

if [ -z "${CONFIG_DIR}" ]; then
    log "Error: CONFIG_DIR is not set"
    exit 1
fi

if [ -z "${PARAM_DIR}" ]; then
    log "Error: PARAM_DIR is not set"
    exit 1
fi

if [ -z "${MODEL_DIR}" ]; then
    log "Error: MODEL_DIR is not set"
    exit 1
fi

# Check if the model directory is not empty
if [ -z "$(ls "${MODEL_DIR}")" ]; then
    log "Model was not found in the volume"
fi

# Start the FastAPI application
log "Starting FastAPI application"
uvicorn app.flat_price_app:app --reload --port "${APP_DOCKER_PORT}" --host 0.0.0.0
log "FastAPI application exited"