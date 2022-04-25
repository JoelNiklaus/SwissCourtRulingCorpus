#!/bin/sh

# This file runs 3 different instances of the same prodigy task.
# Please run setup.sh beforehand.

WARN="\033[1;31m"
NC="\033[0m"

# define usage help
usage=$(cat <<- EOF
  Arguments:
    - task:        type of task (e.g. facts-annotation)
  Usage:
    -bash run.sh task
  Example:
    - bash run.sh facts-annotation

  Consult /prodigy/readme.md for more information.
EOF
)

# parse options
task=$1

if [ -z "$task" ] ; then
  printf "${WARN}Invalid arguments given.\n\n${NC}${usage}\n"
  exit 1
fi

if [ "$(docker ps -q -f name=prodigy_v1_nina)" ]; then
  case "$task" in
    "facts-annotation")
      docker exec -it -d prodigy_v1_nina prodigy "$task" de -F ./recipes/facts_annotation.py
      docker exec -it -d prodigy_v1_nina prodigy "$task" fr -F ./recipes/facts_annotation.py
      docker exec -it -d prodigy_v1_nina prodigy "$task" it -F ./recipes/facts_annotation.py
      printf "${SUCCESS}Starting command in container, to stop use Ctrl+C.\n${NC}"
      ;;
    *)
      printf "${WARN}Unknown task '$task' given.\n\n${NC}${usage}\n"
      exit 1
      ;;
  esac
else
  printf "${WARN}No container with the name prodigy_v1_nina found.\n Use 'bash setup.sh to start it.'${NC}"
fi
