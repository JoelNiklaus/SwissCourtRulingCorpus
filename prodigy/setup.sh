#!/bin/sh

echo "Setup started..."

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

# if the container already exists, remove it
<<comment
if [ "$(docker ps -q -f name=prodigy_v1)" ]; then
  printf "${WARN}A container with the name prodigy_v1 already exists. To use a newer image it has to be removed.\n"
  read -p "Do you want to remove it and use the new image? [y/N]" -n 1 -r
  printf "${NC}\n" # writes a new line


  if [[ $REPLY =~ ^[Yy]$ ]]
  then
      echo "Removing container..."
      docker rm -f prodigy_v1
      echo "Building image..."
      docker build -t prodigy:v1.0 .
  else
      printf "${NC}Exiting...\n"
      exit 1
  fi
fi
comment
echo "Starting container in idle mode..."

docker run -d \
  --name prodigy_v1_nina \
  --network="host" \
  -e DB_USER=$DB_USER \
  -e DB_PASSWORD=$DB_PASSWORD \
  prodigy_v1_nina:latest

printf "${SUCCESS}Setup finished, use 'bash run.sh' to start the server.${NC}\n\n\n"
