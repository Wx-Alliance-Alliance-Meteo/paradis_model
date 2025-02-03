#!/bin/bash -l

#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --job-name=test_rk4
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
end_date="2010-12-31" 
max_epoch=10
forecast_steps=4
initial_LR=1.0e-3
num_workers=8
num_gpu=2
hidden_multiplier=4
method=rk4
echo forecast steps = , ${forecast_steps},  num_works = , $num_workers, num_gpu, $num_gpu, hidden_multiplier, $hidden_multiplier

output_dir="./testoutput-vicky/compare_rk4_fe"
mkdir -p $output_dir
output_file=${method}_from_${start_date}_to_${end_date}_${num_workers}_num_worker_${forecast_steps}_forecast_step_${num_gpu}_gpu_${hidden_multiplier}_hidden_multiplier_${max_epoch}_max_epoch.txt

srun python train.py dataset.training.start_date=$start_date dataset.training.end_date=$end_date dataset.num_workers=$num_workers trainer.max_epochs=$max_epoch model.forecast_steps=$forecast_steps trainer.lr=$initial_LR trainer.num_devices=$num_gpu model.hidden_multiplier=$hidden_multiplier > ${output_dir}/${output_file}












