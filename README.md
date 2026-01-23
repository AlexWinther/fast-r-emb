# FAST-R-Emb: Fast Regression Testing with Embeddings

This project implements budget-aware test suite reduction algorithms using embeddings and traditional methods.

## Local Development

### Setup
```bash
# Install uv (if not already installed)
pip install uv

# Install dependencies
uv sync

# Run budget experiment
uv run python -m budget
```

## HPC Deployment with Singularity/Slurm

### Building the Container

1. **Build Docker image:**
   ```bash
   ./build_docker.sh
   ```

2. **Convert to Singularity SIF (on HPC):**
   ```bash
   singularity build fast-r-emb.sif docker-daemon://fast-r-emb
   ```
   Or pull from registry if pushed to Docker Hub.

### Running on HPC

1. **Transfer data:** Copy your `data/` directory to the HPC system.

2. **Submit Slurm job:** Use the provided `slurm_budget.sh` as a template, adjusting paths and resources as needed.

   Example Slurm script:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=fast-r-emb-budget
   #SBATCH --time=24:00:00
   #SBATCH --mem=32GB
   #SBATCH --cpus-per-task=4

   module load singularity

   singularity exec \
       --bind /path/to/data:/app/data \
       --bind /path/to/output:/app/outputs \
       fast-r-emb.sif \
       uv run python -m budget
   ```

### Directory Structure for HPC

- Mount your data directory to `/app/data` in the container
- Mount an output directory to `/app/outputs` for results
- The container expects test suite data in the format defined in `src/utils.py`

## Dependencies

- Python >= 3.13
- uv (for fast dependency management)
- PyTorch >= 2.9.1
- Transformers >= 4.57.3
- Other ML/data science libraries (see `pyproject.toml` and `uv.lock`)

## Project Structure

- `src/`: Source code
- `data/`: Test suite data (SIR and D4J datasets)
- `outputs/`: Generated results and plots