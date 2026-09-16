#!/bin/bash
# The job that produced the grids in this directory. Submit from the repository
# root, not from here: srun calls the driver by a path relative to it.
#
#   sbatch official_experiments/run_slurm.sh test BPI_Challenge_2012
#   sbatch official_experiments/run_slurm.sh train Sepsis_Case all
#   sbatch official_experiments/run_slurm.sh train Sepsis_Case \
#       baseline marking gnn seq --no-net-eval --out-dir noise_curve_train_features
#
# 16 cores for one GPU: training is light, pm4py token replay is the dominant
# cost and runs on the CPU. gpu80g stays even where 80 GB is not needed, to bind
# every cell to the same card: the timing columns end up in one table.

#SBATCH --job-name=nspm
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpu-low
#SBATCH --account=ppm-sk
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80g
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20-00:00:00
#SBATCH --requeue

set -euo pipefail

PROTOCOL="${1:?first argument: test or train}"
DATASET="${2:?second argument: the directory name under datasets/}"
shift 2

# Variants first, then flags for noise_curve.py. The boundary is the first token
# starting with "--", which no method name ever does.
VARIANTS=()
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --*) EXTRA=("$@"); break ;;
    *)   VARIANTS+=("$1"); shift ;;
  esac
done

if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(lll gll)
elif [ ${#VARIANTS[@]} -eq 1 ] && [ "${VARIANTS[0]}" = "all" ]; then
  VARIANTS=(baseline checker checker_net checker_net_state marking gnn seq lll gll)
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source /opt/share/tools/anacoda/2024.06-1/etc/profile.d/conda.sh
# torch in nspm is built for sm_75 and up: it runs on the A100s, not on the
# sm_70 V100s, which need the nspm-v100 clone chosen at launch:
#   sbatch --export=ALL,NSPM_ENV=$HOME/envs/nspm-v100 -C gpu32g run_slurm.sh ...
conda activate "${NSPM_ENV:-$HOME/envs/nspm}"
echo "environment $(python -c 'import torch,sys; print(sys.prefix, torch.__version__)')"

# The code halves cpu_count() because locally that counts SMT threads;
# SLURM_CPUS_PER_TASK already counts cores.
export NSPM_REPLAY_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
# BLAS opens one thread per core in every child of the pool: without this,
# 16 workers x 16 threads bring the node down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== $(date) ==="
echo "protocol $PROTOCOL | dataset $DATASET | variants ${VARIANTS[*]}"
echo "extra flags: ${EXTRA[*]:-none}"
echo "$NSPM_REPLAY_WORKERS workers | node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

# nvidia-smi seeing the card is not enough: torch has failed to initialise CUDA
# inside the srun step anyway, and resolve_device("auto") then falls back to the
# CPU silently at twenty times the cost -- 69 Sepsis cells in 15h, not 3h.
srun python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch sees the GPU:', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { echo "ABORT: torch does not see the GPU, training would run on the CPU"; exit 1; }

# Under set -u an empty array expanded directly is an error, and launches
# without flags are the norm.
srun python official_experiments/scripts/noise_curve.py \
    --protocol "$PROTOCOL" \
    --dataset "$DATASET" \
    --variants "${VARIANTS[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "=== done $(date) ==="
