#!/bin/sh

# This file is intended to help develop the prodigy application.
# It runs a shell in the container, with all files mounted from the host to the container, allowing
# edits of all files in /prodigy to be made without restarting the container.
# Do not use this to actually run the server as it might corrupt a running operation.

# To develop the facts_annotation task please the following command
# prodigy facts-annotation de "" test -F recipes/facts_annotation.py
# This runs the prodigy task on the test port, uses a test set as input and saves the annotation as a test set.
# Please do not run the recipe in the productive mode because the datasets are currently being annotated by the legal experts.

# fetch env variables
if test -f .env; then
    source .env
else
    echo "No .env file found. Please create one containing DB_USERNAME and DB_PASSWORD."
    exit 1
fi

# start a container with a prefix for the current user
# mount the /prodigy directory from the host to the container
echo "Starting development container..."
docker run -d \
  --name dev_"$USER"_prodigy \
  --network="host" \
  -e DB_USER=$DB_USER \
  -e DB_PASSWORD=$DB_PASSWORD \
  --mount type=bind,source="$(pwd)",target=/app \
  prodigy_v1_nina:latest

printf "Starting bash shell in development container...\nUse 'exit' to exit the shell."
docker exec -it dev_"$USER"_prodigy bash
# stop and remove the container after the session, to prevent a pileup of unused containers
echo "Cleanup, this might a few seconds..."
docker stop dev_"$USER"_prodigy
docker container rm dev_"$USER"_prodigy
echo "Done."