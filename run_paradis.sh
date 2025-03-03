#!/bin/bash -l

#SBATCH --job-name=compare_rk4_fe
#SBATCH --output=./testoutput-vicky/%x-%j.out
#SBATCH --error=./testoutput-vicky/%x-%j.err
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#SBATCH --account=eccc_mrd__gpu_a100
#SBATCH --partition=gpu_a100                         # For A100 GPUs on GPSC7, use gpu_a100 partition name
#SBATCH --time=0-20:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=33
#SBATCH --gpus=2
#SBATCH --mem=200G
#SBATCH --export=ALL

nvidia-smi
# activate conda env
conda activate /home/cap003/hall5/software/miniconda3/envs/gatmosphere

start_date="2010-01-01" 
end_date="2019-12-31" 
val_start="1979-01-01"
val_end="1980-12-31"
max_epoch=500
forecast_steps=1
initial_LR=1.0e-3
scheduler="reduced_lr"
scheduler_factor=0.75
scheduler_patience=5
scheduler_threshold=1.e-4 
scheduler_threshold_mode="rel" 
scheduler_min_lr=1.e-7 
num_workers=8
num_gpu=2
hidden_multiplier=4
integrator="fe"
train_dt=21600
compile=false
batch_size=64
num_substeps=6
early_stop=false

output_dir="./testoutput-vicky/compare_rk4_fe"
mkdir -p $output_dir
output_file=${integrator}_from_${start_date}_to_${end_date}_${num_workers}_num_worker_traindt_${train_dt}_with_${num_substeps}_substeps_${hidden_multiplier}_hidden_multiplier_${max_epoch}_max_epoch.txt

echo $output_file 
echo "
start_date= $start_date
end_date=$end_date
val_start=$val_start
val_end=$val_end
max_epoch=$max_epoch
forecast_steps=$forecast_steps
initial_LR=$initial_LR
scheduler=$scheduler
scheduler_factor=$scheduler_factor
scheduler_patience=$scheduler_patience
scheduler_threshold=$scheduler_threshold
scheduler_threshold_mode=$scheduler_threshold_mode
scheduler_min_lr=$scheduler_min_lr
num_workers=$num_workers
num_gpu=$num_gpu
hidden_multiplier=$hidden_multiplier
integrator=$integrator
train_dt=$train_dt
compile=$compile
batch_size=$batch_size
num_substeps=$num_substeps
"

srun python train.py \
	training.dataset.start_date=$start_date \
	training.dataset.end_date=$end_date \
	training.validation_dataset.start_date=$val_start \
	training.validation_dataset.end_date=$val_end \
	compute.num_workers=$num_workers \
	compute.integrator=$integrator \
	training.parameters.max_epochs=$max_epoch \
	model.base_dt=$train_dt \
	model.forecast_steps=$forecast_steps \
	model.num_substeps=$num_substeps \
	training.parameters.lr=$initial_LR \
    	training.parameters.scheduler.factor=$scheduler_factor \
    	training.parameters.scheduler.patience=$scheduler_patience \
    	training.parameters.scheduler.threshold=$scheduler_threshold \
    	training.parameters.scheduler.threshold_mode=$scheduler_threshold_mode \
    	training.parameters.scheduler.min_lr=$scheduler_min_lr \
	compute.num_devices=$num_gpu \
	model.hidden_multiplier=$hidden_multiplier \
	compute.batch_size=$batch_size > ${output_dir}/${output_file}












