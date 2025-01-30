#!/bin/bash -l

#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=test_rk_rk4_2phase_combined_4_workers_1gpu
#SBATCH --output=./testoutput-vicky/%x-%j.out
#SBATCH --error=./testoutput-vicky/%x-%j.err                       # jobname-jobid.err naming
#SBATCH --account=eccc_mrd__gpu_a100
#SBATCH --partition=gpu_a100                         # For A100 GPUs on GPSC7, use gpu_a100 partition name
#SBATCH --time=0-10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=33
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --export=ALL

nvidia-smi
# activate conda env
conda activate /home/cap003/hall5/software/miniconda3/envs/gatmosphere

start_date="2010-01-01" 
end_date="2019-12-31" 
max_epoch=1000
forecast_steps=4
initial_LR=1.0e-3
num_workers=4
num_gpu=1
hidden_multiplier=4

echo forecase steps = , ${forecast_steps},  num_works = , $num_workers, num_gpu, $num_gpu, hidden_multiplier, $hidden_multiplier

srun python train.py dataset.training.start_date=$start_date dataset.training.end_date=$end_date dataset.num_workers=$num_workers trainer.max_epochs=$max_epoch model.forecast_steps=$forecast_steps trainer.lr=$initial_LR trainer.num_devices=$num_gpu model.hidden_multiplier=$hidden_multiplier












