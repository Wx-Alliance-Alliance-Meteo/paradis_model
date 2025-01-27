#!/bin/bash -l

#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=test_rk_fe_2phases_combined
#SBATCH --output=./testoutput-vicky/%x-%j.out
#SBATCH --error=./testoutput-vicky/%x-%j.err                       # jobname-jobid.err naming
#SBATCH --account=eccc_mrd__gpu_a100
#SBATCH --partition=gpu_a100                         # For A100 GPUs on GPSC7, use gpu_a100 partition name
#SBATCH --time=0-10:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=33
#SBATCH --gpus=2
#SBATCH --mem=120G
#SBATCH --export=ALL

nvidia-smi
# activate conda env
conda activate /home/cap003/hall5/software/miniconda3/envs/gatmosphere

start_date="2010-01-01" 
end_date="2019-12-31" 
max_epoch=1000
forecast_steps=4
initial_LR=5e-5 
fe_ckpt_path="/home/siw001/paradis_model/logs/lightning_logs/version_fe_phase1/checkpoints/best.ckpt"
rk4_ckpt_path="/home/siw001/paradis_model/logs/lightning_logs/version_rk4_phase1/checkpoints/best.ckpt"
num_workers=4

#print("forecase steps = ", ${forecast_steps}, " num_works = ", $num_workers. "\n")
srun python train.py dataset.start_date=$start_date dataset.end_date=$end_date dataset.num_workers=$num_workers trainer.max_epochs=$max_epoch model.forecast_steps=$forecast_steps 

#trainer.lr=$initial_LR model.checkpoint_path=$rk4_ckpt_path











