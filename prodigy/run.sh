#!/bin/sh

WARN="\033[1;31m"
NC="\033[0m"

# define usage help
usage=$(cat <<- EOF
  Arguments:
    - task:        type of task (facts-annotation)

  Usage:
    -bash run.sh task
  Example:
    - bash run.sh facts-annotation

  Consult /prodigy/README.md for more information.
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
      printf "${SUCCESS}Starting command in container, to stop use Ctrl+C.\n${NC}"
      docker exec -it -d prodigy_v1_nina prodigy "$task" de 1 "" -F ./recipes/facts_annotation.py
      docker exec -it -d prodigy_v1_nina prodigy "$task" de 2 "" -F ./recipes/facts_annotation.py
      docker exec -it -d prodigy_v1_nina prodigy "$task" de 3 "" -F ./recipes/facts_annotation.py
      docker exec -it -d prodigy_v1_nina prodigy "$task" fr 1 "" -F ./recipes/facts_annotation.py
      docker exec -it -d prodigy_v1_nina prodigy "$task" it 1 "" -F ./recipes/facts_annotation.py
      ;;
    *)
      printf "${WARN}Unknown task '$task' given.\n\n${NC}${usage}\n"
      exit 1
      ;;
  esac
else
  printf "${WARN}No container with the name prodigy_v1_nina found.\n Use 'bash setup.sh to start it.'${NC}"
fi
