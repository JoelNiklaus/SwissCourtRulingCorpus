#!/bin/sh

WARN="\033[1;31m"
NC="\033[0m"

if [ "$(docker ps -q -f name=prodigy_v1)" ]; then
  docker exec -it prodigy_v1 prodigy judgment-outcome dataset_name_1 -F ./recipes/judgement_outcome.py
else
  printf "${WARN}No container with the name prodigy_v1 found.\n Use 'bash setup.sh to start it.'${NC}"
fi