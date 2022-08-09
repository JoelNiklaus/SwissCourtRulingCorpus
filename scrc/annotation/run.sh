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
    "all-tasks")
    for VARIABLE in de fr it
      do
        printf "${SUCCESS}Starting facts-annotation command in container, to stop use Ctrl+C.\n${NC}"
        docker exec -it -d prodigy_v1_nina prodigy facts-annotation $VARIABLE -F ./judgment_explainability/recipes/facts_annotation.py
        printf "${SUCCESS}Starting judgment-prediction command in container, to stop use Ctrl+C.\n${NC}"
        docker exec  -it -d prodigy_v1_nina prodigy judgment-prediction $VARIABLE -F ./judgment_prediction/recipes/judgment_prediction.py
      done
    for VARIABLE in angela lynn thomas
      do
        printf "${SUCCESS}Starting inspect-facts-annotation command in container, to stop use Ctrl+C.\n${NC}"
        docker exec -it -d prodigy_v1_nina prodigy inspect-facts-annotation de $VARIABLE -F ./judgment_explainability/recipes/inspect_facts_annotation.py
      done
    printf "${SUCCESS}Starting inspect-facts-annotation command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy inspect-facts-annotation fr lynn -F ./judgment_explainability/recipes/inspect_facts_annotation.py
    printf "${SUCCESS}Starting inspect-facts-annotation command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy inspect-facts-annotation it lynn -F ./judgment_explainability/recipes/inspect_facts_annotation.py
    printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy "$task" gold_annotations_de annotations_de-angela,annotations_de-lynn,annotations_de-thomas -l "Supports judgment","Opposes judgment","Lower court","Neutral" -v spans_manual --auto-accept

    ;;
    "facts-annotation")
    for VARIABLE in de fr it
      do
        printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
        docker exec -it -d prodigy_v1_nina prodigy "$task" $VARIABLE -F ./judgment_explainability/recipes/facts_annotation.py
      done
      ;;
    "inspect-facts-annotation")
    printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy "$task" fr lynn -F ./judgment_explainability/recipes/inspect_facts_annotation.py
    for VARIABLE in angela lynn thomas
      do
        printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
        docker exec -it -d prodigy_v1_nina prodigy "$task" de $VARIABLE -F ./judgment_explainability/recipes/inspect_facts_annotation.py
      done
    ;;

  "review")
    printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy "$task" gold_annotations_de annotations_de-angela,annotations_de-lynn,annotations_de-thomas -l "Supports judgment","Opposes judgment","Lower court","Neutral" -v spans_manual --auto-accept
    ;;
    "judgment-prediction")
  for VARIABLE in de fr it
    do
      printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
      docker exec  -it -d prodigy_v1_nina prodigy "$task" $VARIABLE -F ./judgment_prediction/recipes/judgment_prediction.py
    done
  ;;

  "db-out")
  for VARIABLE in annotations_de annotations_de-angela annotations_de-lynn annotations_de-thomas annotations_fr annotations_fr-lynn annotations_it-angela annotations_it annotations_de_inspect annotations_de_inspect-lynn annotations_de_inspect-thomas

    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE > ./judgment_explainability/annotations/$VARIABLE.jsonl
    done
   ;;
  "drop")
  for VARIABLE in annotations_de-nina annotations_de-ninaa annotations_de-ninaaa annotations_it-nina annotations_fr-nina gold_annotations_de-nina gold_annotations_de-test

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
