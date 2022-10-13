#!/bin/sh

WARN="\033[1;31m"
SUCCESS="\033[1;32m"
NC="\033[0m"

# fetch env variables
if test -f .env; then
    source .env
else
    echo "No .env file found. Please create one containing DB_USER and DB_PASSWORD."
    exit 1
fi
echo "Stopping and removing container prodigy_nina"
docker stop prodigy_nina
docker rm prodigy_nina
docker image rm prodigy_nina:latest
echo "Building image"
docker build \
  --build-arg user=${PRODIGY_BASIC_AUTH_USER} \
  --build-arg password=${PRODIGY_BASIC_AUTH_PASS} \
  -t prodigy_nina .

echo "Setup started..."

echo "Starting container prodigy_nina in idle mode..."
docker run -d \
  --restart unless-stopped \
  --name prodigy_nina \
  --network="host" \
  -e DB_USER="$DB_USER" \
  -e DB_PASSWORD="$DB_PASSWORD" \
  --mount type=bind,source="$(pwd)",target=/app \
  prodigy_nina:latest

printf "${SUCCESS}Setup finished, use 'bash run.sh' to start the server.${NC}\n\n\n"
