#!/bin/sh

WARN="\033[1;31m"
NC="\033[0m"

# define usage help
usage=$(cat <<- EOF
  Arguments:
    - task:        type of task (judgment-outcome extraction, TBD)
    - language:    the language to use for the server, e.g. 'de'
    - spider_name: the name of the spider to use, e.g. 'BGer'

  Usage:
    ./run.sh task language spider_name

  Example:
    ./run.sh judgment-outcome de BGer

  Consult /prodigy/README.md for more information.
EOF
)

# parse options
task=$1
language=$2
spider_name=$3

if [ -z "$task" ] || [ -z "$language" ] || [ -z "$spider_name" ]; then
  printf "${WARN}Invalid arguments given.\n\n${NC}${usage}\n"
  exit 1
fi

if [ "$(docker ps -q -f name=prodigy_v1)" ]; then
  case "$task" in
    "judgment-outcome")
      printf "${SUCCESS}Starting command in container, to stop use Ctrl+C.\n${NC}"
      docker exec -it prodigy_v1 prodigy "$task" "$spider_name" "$language" -F ./recipes/judgment_outcome.py
      ;;
    *)
      printf "${WARN}Unknown task '$task' given.\n\n${NC}${usage}\n"
      exit 1
      ;;
  esac
else
  printf "${WARN}No container with the name prodigy_v1 found.\n Use 'bash setup.sh to start it.'${NC}"
fi
