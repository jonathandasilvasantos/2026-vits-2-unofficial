#!/bin/bash
# resume.sh - Resume VITS-2 training from last checkpoint
# Uses paper-standard parameters (accum=1, same LR, no R1, no grad clip)
#
# Usage:
#   ./resume.sh                    # resume from latest checkpoint
#   ./resume.sh --skip-tests       # skip unit tests (faster restart)
#
# To run inside tmux "shared" session:
#   tmux send-keys -t shared './resume.sh --skip-tests' Enter

CHECKPOINT="checkpoints/vits2_latest.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -la checkpoints/*.pt 2>/dev/null
    exit 1
fi

docker run --gpus all --rm \
    -v /home/bender/projects/vits-2:/workspace \
    -w /workspace \
    --shm-size=8g \
    vits2 \
    python train.py \
        --config config.json \
        --checkpoint "$CHECKPOINT" \
        "$@" \
    2>&1 | tee /home/bender/projects/vits-2/training.log
