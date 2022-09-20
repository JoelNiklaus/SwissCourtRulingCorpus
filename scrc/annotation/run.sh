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
    printf "${SUCCESS}Starting review command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy review gold_annotations_de annotations_de-angela,annotations_de-lynn,annotations_de-thomas -l "Supports judgment","Opposes judgment","Lower court","Neutral" -v spans_manual --auto-accept --show-skipped

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
    printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy "$task" it lynn -F ./judgment_explainability/recipes/inspect_facts_annotation.py
    for VARIABLE in angela lynn thomas
      do
        printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
        docker exec -it -d prodigy_v1_nina prodigy "$task" de $VARIABLE -F ./judgment_explainability/recipes/inspect_facts_annotation.py
      done
    ;;

  "review-de")
    printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy review gold_annotations_de annotations_de_inspect-lynn,gold_annotations_de-gold_final_v3 -l "Supports judgment","Opposes judgment","Lower court","Neutral" -v spans_manual --show-skipped
    ;;
  "review-fr")
    printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy review gold_annotations_fr annotations_fr_inspect-lynn -l "Supports judgment","Opposes judgment","Lower court","Neutral" -v spans_manual --show-skipped
    ;;
  "review-it")
    printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
    docker exec -it -d prodigy_v1_nina prodigy review gold_annotations_it annotations_it-lynn,gold_annotations_it-gold_nina -l "Supports judgment","Opposes judgment","Lower court","Neutral" -v spans_manual --show-skipped
    ;;
    "judgment-prediction")
  for VARIABLE in de fr it
    do
      printf "${SUCCESS}Starting $task command in container, to stop use Ctrl+C.\n${NC}"
      docker exec  -it -d prodigy_v1_nina prodigy "$task" $VARIABLE -F ./judgment_prediction/recipes/judgment_prediction.py
    done
  ;;

  "db-out")
  for VARIABLE in annotations_de annotations_de-angela annotations_de-lynn annotations_de-thomas annotations_de_inspect annotations_de_inspect-lynn annotations_de_inspect-thomas
    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE > ./judgment_explainability/annotations/de/$VARIABLE.jsonl
    done
    for VARIABLE in gold_annotations_de gold_annotations_de-gold_final
    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE > ./judgment_explainability/annotations/de/gold/$VARIABLE.jsonl
    done
    for VARIABLE in annotations_fr annotations_fr-lynn annotations_fr_inspect annotations_fr_inspect-lynn
    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE > ./judgment_explainability/annotations/fr/$VARIABLE.jsonl
    done
        for VARIABLE in gold_annotations_fr gold_annotations_fr-gold_nina
    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE > ./judgment_explainability/annotations/fr/gold/$VARIABLE.jsonl
    done
    for VARIABLE in annotations_it annotations_it-lynn annotations_it-angela annotations_it_inspect annotations_it_inspect-lynn
    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE > ./judgment_explainability/annotations/it/$VARIABLE.jsonl
    done
        for VARIABLE in gold_annotations_it gold_annotations_it-gold_nina
    do
      docker exec prodigy_v1_nina prodigy "$task" $VARIABLE > ./judgment_explainability/annotations/it/gold/$VARIABLE.jsonl
    done
   ;;

  "drop")
  for VARIABLE in
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
