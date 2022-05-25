#!/bin/sh

# This file runs 3 different instances of the same prodigy task.
# Please run setup.sh beforehand.

WARN="\033[1;31m"
NC="\033[0m"

# define usage help
usage=$(cat <<- EOF
  Arguments:
    - task:        type of task (facts-annotation, inspect-facts-annotation, review, db-out, drop, stats)
  Usage:
    -bash run.sh task
  Example:
    - bash run.sh facts-annotation

  Consult readme.md for more information.
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
    for VARIABLE in de fr it
      do
        docker exec -it -d prodigy_v1_nina prodigy "$task" $VARIABLE -F ./recipes/facts_annotation.py
      done
      printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
      ;;


    "inspect-facts-annotation")
    for VARIABLE in angela lynn thomas
      do
        docker exec -it -d prodigy_v1_nina prodigy "$task" de $VARIABLE -F ./recipes/inspect_facts_annotation.py
      done
    printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
    ;;

  "review")
    docker exec -it -d prodigy_v1_nina prodigy "$task" gold_annotations_de annotations_de-angela,annotations_de-lynn,annotations_de-thomas -l "Supports judgment","Opposes judgment","Lower court" -v spans_manual --auto-accept
    printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
    ;;

  "db-out")
  for VARIABLE in annotations_de annotations_de-angela annotations_de-lynn annotations_de-thomas
    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE > ./annotations/$VARIABLE.jsonl
    done
   ;;
  "drop")
  for VARIABLE in annotations_de-nina annotations_de-ninaa annotations_de-ninaaa
    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE
    done
   ;;

  "stats")
  for VARIABLE in annotations_de-angela annotations_de-lynn annotations_de-thomas
    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE
    done
  ;;

    *)
      printf "${WARN}Unknown task '$task' given.\n\n${NC}${usage}\n"
      exit 1
      ;;
  esac
else
  printf "${WARN}No container with the name prodigy_v1_nina found.\n Use 'bash setup.sh to start it.'${NC}"
fi
