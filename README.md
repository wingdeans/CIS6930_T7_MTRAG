# `CIS6930_T7_MTRAG`

Fine-tuned Llama models on the MTRAG benchmark

## Getting Started
Training:
```bash
python finetune.py --modelname $MODEL_NAME --raft $RAFT --cot $COT \
    --epochs 11 --lr 3e-4 --batchsize 1 --gradsteps 4 \
    --contextlength 8192 --lorarank 8 --loraalpha 32
```

Evaluation:
```bash
uv venv
source .venv/bin/activate
python3 mt-rag-benchmark/scripts/evaluation/run_generation_eval.py \
    -i CIS6930_T7_MTRAG/outputs/$INPUT_FILE \
    -o $OUTPUT_FILE \
    -e mt-rag-benchmark/scripts/evaluation/config.yaml \
    --provider hf --judge_model ibm-granite/granite-3.3-8b-instruct
```

Aggregate:
```bash
uv run agg_metrics.py --input_file $OUTPUT_FILE --modelname $MODEL_NAME --raft $RAFT --cot $COT
```

<details>
  <summary>Legacy conda installation</summary>
The recommendation is to create two conda environments; one for training/inference, and one for evaluation

When you've installed the conda environments as below, repoint the conda environments in `slurm_job.sh` (training and inference) and `slurm_job_eval.sh` (evaluation) to the names of your conda environments in your /home directory

Then you're ready to kick off these slurm jobs on HiperGator. Note that the code currently uses my HuggingFace token to access and download the Llama models from HF (in `hf_login.py`)

### Training conda environment

Create and activate a conda environment (using python 3.11):
```
conda create -p mtraghome python=3.11
conda activate ./mtraghome
```

Install via pip in a conda env for training and inference (tested on python 3.11). This can take up ~15GB space in your /home directory:
```
pip install llama-cookbook
```

### Inference conda environment
Install the below in a separate conda env for evaluation (tested on python 3.11):
- `torch==2.8.0`
- the `requirements.txt` file in `mt-rag-benchmark/scripts/evaluation`
Note, use as many cores as possible when installing flash-attn. I used 16 cores on hipergator with 32GB ram. A sample slurm script has been included in this folder `sample_flash_attn_install_script.sh`
</details>

