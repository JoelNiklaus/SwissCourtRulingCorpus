#!/bin/sh

WARN="\033[1;31m"
SUCCESS="\033[1;32m"
NC="\033[0m"

# fetch env variables
if test -f .env; then
    source .env
else
    echo "No .env file found. Please create one containing DB_USERNAME and DB_PASSWORD."
    exit 1
fi
echo "Stopping and removing container prodigy_v1_nina "
docker stop prodigy_v1_nina
docker rm prodigy_v1_nina
docker image rm prodigy_v1_nina:latest
echo "Building image"
docker build \
  --build-arg user=${PRODIGY_BASIC_AUTH_USER} \
  --build-arg password=${PRODIGY_BASIC_AUTH_PASS} \
  -t prodigy_v1_nina .