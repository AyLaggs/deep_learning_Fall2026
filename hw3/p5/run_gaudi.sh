#!/bin/bash
# Slurm requests one node and one Gaudi accelerator for the single-device run.
#SBATCH -A class_eee598fall2026
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -c 18
#SBATCH -t 0-23
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -o gaudi_output.%j.out
#SBATCH -e gaudi_output.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alaguda@asu.edu
#SBATCH --export=NONE

# Load shell initialization before enabling nounset: the cluster's /etc/bashrc
# may reference BASHRCSOURCED before initializing it.
source ~/.bashrc

# mamba/latest is a Lua modulefile, so use Lmod rather than Tcl Environment
# Modules. Prefer the node's configured LMOD_CMD, then common Lmod init paths.
if [ -n "${LMOD_CMD:-}" ] && [ -x "$LMOD_CMD" ]; then
    eval "$("$LMOD_CMD" bash autoinit)"
elif [ -n "${LMOD_INIT:-}" ] && [ -r "$LMOD_INIT" ]; then
    source "$LMOD_INIT"
elif [ -r /etc/profile.d/lmod.sh ]; then
    source /etc/profile.d/lmod.sh
elif [ -r /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
elif [ -r /packages/lmod/lmod/init/bash ]; then
    source /packages/lmod/lmod/init/bash
else
    echo "Lmod initialization was not found; cannot load mamba/latest.lua." >&2
    echo "Set LMOD_INIT to the cluster's Lmod bash initialization script." >&2
    exit 1
fi

# Stop on failed commands, undefined variables, or failed stages in a pipeline
# (including the training command piped through tee).
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load mamba module (cluster-provided)
module load mamba/latest

# Use the cluster-provided Gaudi environment directly, as recommended by the
# ASU Research Computing guide. Do not layer a virtualenv or pip-installed
# Transformers/Optimum versions over its HPU-enabled PyTorch stack.
source activate gaudi-pytorch-vllm

# Ignore packages left in ~/.local by earlier pip installs. In particular, a
# user-local CPU torch can shadow the cluster's HPU-enabled torch and fail to
# load libtorch_python.so.
export PYTHONNOUSERSITE=1

# Keep temporary files and Hub downloads on node-local scratch instead of the
# home filesystem, which has a limited quota.
JOB_SCRATCH="${SLURM_TMPDIR:-/tmp/${USER:-user}/chartqa-${SLURM_JOB_ID:-manual}}"
mkdir -p "$JOB_SCRATCH/tmp" "$JOB_SCRATCH/huggingface" "$JOB_SCRATCH/matplotlib"
export TMPDIR="$JOB_SCRATCH/tmp"
export HF_HOME="$JOB_SCRATCH/huggingface"
export MPLCONFIGDIR="$JOB_SCRATCH/matplotlib"

# The shared vLLM environment does not include all packages needed for LoRA
# training. Check the complete dependency set and, if needed, create a job-local
# overlay. Optimum Habana 1.19.1 was validated for SynapseAI 1.22 and requires
# Transformers 4.51.x, so those Python packages are installed in the overlay;
# the shared HPU-enabled PyTorch remains untouched. Normal dependency
# resolution supplies transitive requirements such as datasets' Arrow stack.
if ! python -c "import optimum.habana, peft, accelerate, datasets, matplotlib, PIL, sentencepiece, huggingface_hub" >/dev/null 2>&1; then
    python -m venv --system-site-packages "$JOB_SCRATCH/venv"
    source "$JOB_SCRATCH/venv/bin/activate"
    python -m pip install --quiet --upgrade-strategy only-if-needed \
        "optimum-habana==1.19.1" \
        "transformers==4.51.3" \
        "peft==0.12.0" \
        "accelerate==1.10.1" \
        "datasets>=4.0.0" \
        "matplotlib" \
        "pillow" \
        "sentencepiece" \
        "huggingface_hub>=0.26.0,<1.0"
fi

# Verify the HPU stack and training dependencies before downloading model
# assets or starting the long baseline run.
python -c "import torch, transformers, peft, datasets, matplotlib, accelerate, importlib.metadata as metadata; from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments; from accelerate.utils import FP8RecipeKwargs; print('PyTorch:', torch.__version__, torch.__file__); print('Transformers:', transformers.__version__); print('Optimum Habana:', metadata.version('optimum-habana')); print('Accelerate:', accelerate.__version__); print('PEFT:', peft.__version__); print('Datasets:', datasets.__version__)"

# Export Hub IDs for the download commands and Python training process.
export MODEL_ID="llava-hf/llava-v1.6-mistral-7b-hf"
export DATASET_ID="HuggingFaceM4/ChartQA"

# Use eager execution mode for the example's reliable Gaudi configuration.
export PT_HPU_LAZY_MODE=0

# Print interpreter details to help diagnose environment selection on the node.
echo "Python:"
which python
python --version
echo "Scratch directory: $JOB_SCRATCH"
df -h "$JOB_SCRATCH" "$SUBMIT_DIR"

# Fail before downloading/training if the environment cannot access the HPU.
echo "Testing Habana environment..."
python -c "import habana_frameworks.torch.core; import torch; print('Habana import: OK'); print('torch:', torch.__version__); print('HPU available:', torch.hpu.is_available())"
python -c "import torch; assert torch.hpu.is_available(), 'No Gaudi HPU detected'"

# Pre-fetch weights and dataset shards into HF_HOME. The Python program then
# loads the same Hub IDs from this cache instead of downloading them again.
echo "Downloading model from Hugging Face: $MODEL_ID"
python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["MODEL_ID"])'

echo "Downloading ChartQA from Hugging Face: $DATASET_ID"
python -c 'import os; from datasets import load_dataset; load_dataset(os.environ["DATASET_ID"])'

# Run full training/evaluation and retain a copy of stdout/stderr in the job log.
echo "Starting LLaVA-1.6 LoRA fine-tuning on ChartQA..."
python my_llava.py \
    --model_name_or_path "$MODEL_ID" \
    --dataset_name "$DATASET_ID" \
    --output_dir "${OUTPUT_DIR:-./output_llava_chartqa}" \
    2>&1 | tee my_llava.log