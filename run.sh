#!/bin/bash

# Check that miniconda3 is installed
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: miniconda3 is not installed." >&2
    exit 1
fi

# Activate the torch-env environment
source ~/miniconda3/bin/activate torch-env || {
    echo "Error: Failed to activate torch-env environment." >&2
    exit 1
}

# Set constants and variables
Q_VALUES=(far_point far_cluster)
EPS_VALUES=(0 0.05 0.1 0.2)
R_VALUES=(1 5 10 20)
S_VALUES=(40 80 120 160 200)
SEEDS_PER_GPU=3

ROOT="$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")" # save the project root folder to be where the run script at

cd "$ROOT"/experiments || {
    echo "failed to navigate to experiments folder"
    exit 1
}

# Loop over each eps value and train the model for each seed
for Q in "${Q_VALUES[@]}"; do
  mkdir -p "$Q"
  cd "$Q" || {
          echo "Error: Failed to navigate to directory." >&2
          exit 1
      }
  for EPS in "${EPS_VALUES[@]}"; do
    mkdir -p "EPS${EPS}"
    cd "EPS${EPS}" || {
            echo "Error: Failed to navigate to directory." >&2
            exit 1
        }
    for R in "${R_VALUES[@]}"; do
      mkdir -p "R${R}"
      cd "R${R}" || {
              echo "Error: Failed to navigate to directory." >&2
              exit 1
          }
      for S in "${S_VALUES[@]}"; do
        mkdir -p "S${S}"
        cd "S${S}" || {
                echo "Error: Failed to navigate to directory." >&2
                exit 1
            }


            mkdir -p logs
            SEED=0
            BATCH=0
            while [ "$SEED" -lt 14 ]; do
                    START=$((SEEDS_PER_GPU * BATCH))
                    END=$((START + SEEDS_PER_GPU - 1))
                    if [ "$SEED" -lt 15 ]; then
                        printf "Training model for seeds %s-%s on GPU %s...\n" "$START" "$END" "$GPU"
                        for SEED in $(seq "$START" "$END"); do
                            python -u "$ROOT"/train.py \
                                --eps "$EPS" \
                                --Q "$Q" \
                                --r "$R" \
                                --k "$S" \
                                --seed "$SEED" \
                                > "logs/seed$SEED.log" &
                        done
                    fi
                BATCH=$((BATCH+1))
                wait
        done
        cd ..
      done
      cd ..
    done
    cd ..
  done
  cd ..
done

echo "Training complete."