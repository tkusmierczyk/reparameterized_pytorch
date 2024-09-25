#!/bin/bash
#SBATCH --job-name=<JOB_NAME>        # Job name
#SBATCH --output=<SBATCH_DIR>/<JOB_NAME>_output_%j.log   # Output file name (%j will be replaced by the job ID)
#SBATCH --error=<SBATCH_DIR>/<JOB_NAME>_error_%j.log     # Error file name (%j will be replaced by the job ID)
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=6            # Number of CPU cores per task
#SBATCH --mem=24G                    # Total memory for the job
#SBATCH --gres=gpu:1                    
#SBATCH --partition=<PARTITION>      # Partition name

## Load necessary modules
##module load python/3.x               # Replace with your required Python module

# Activate the .env environment
source "<ENV_PATH>/bin/activate"

# Run the Python script
<COMMAND>