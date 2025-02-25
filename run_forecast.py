#!/bin/bash -l

#SBATCH --job-name=forecast_paradis
#SBATCH --output=./testoutput-vicky/forecast/%x-%j.out
#SBATCH --error=./testoutput-vicky/forecast/%x-%j.err                       # jobname-jobid.err naming
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

forecast_start_date="2018-01-01" 
forecast_steps=6
output_freq=1
forecast_dt=21600
num_gpu=2
hidden_multiplier=4
method=rk4
train_dt=6hr
forcing="forcing" 
forecast_data_time_resol=6h

forecast_inf=forecast_with_${method}_train_${train_dt}_forecast_${forecast_dt}s_${forcing}

echo $forecast_inf 

python forecast.py \
        model.checkpoint_path=/home/siw001/hall6/paradis_model_fe/logs/lightning_logs/version_fe_8yr_6hr_1step_500epoch/checkpoints/best.ckpt \
        forecast.start_date=${forecast_start_date} \
        model.forecast_steps=${forecast_steps} \
        model.base_dt=${forecast_dt} \
        forecast.output_frequency=${output_freq} \
        dataset.time_resolution=${forecast_data_time_resol} \
        --config-name=paradis_settings_forecast

#logs/lightning_logs/version_rk4_8yr_${train_dt}_1step_forcing_500epoch/checkpoints/best.ckpt 

#cd results
#zip /home/siw001/hall6/paradis_model_rk4/testoutput-vicky/forecast_plots/${forecast_inf}.zip *
#zip /home/siw001/hall6/paradis_model_rk4/testoutput-vicky/forecast_plots/${forecast_inf}_data.zip */*/*
#rm -rf *









