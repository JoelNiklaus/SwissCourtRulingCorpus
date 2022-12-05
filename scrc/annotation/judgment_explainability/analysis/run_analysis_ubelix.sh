#!/bin/bash
#SBATCH --job-name="analysis_nina"
#SBATCH --mail-user=nina.baumgartner@unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --qos=job_gpu_preempt
#SBATCH --partition=gpu
#SBATCH --array=2-4

eval "$(conda shell.bash hook)"
conda activate judgement_explainability

for VARIABLE_1 in de fr it;
      do
        python annotation_analysis.py $VARIABLE_1
        python occlusion_analysis.py $VARIABLE_1
        python lower_court_analysis.py $VARIABLE_1
      done;

# run with: sbatch run_analysis.sh