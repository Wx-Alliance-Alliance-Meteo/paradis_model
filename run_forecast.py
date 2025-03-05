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

method=rk4
forecast_path="version_rk4_10yr_2substep_reduce_lr_300epoch"
substep=6
 

python forecast.py \
        model.checkpoint_path=/home/siw001/hall6/paradis_model_rk4/logs/lightning_logs/${forecast_path}/checkpoints/best.ckpt \
        compute.integrator=$method \
        model.num_substeps=$substep \
        --config-name=paradis_settings_forecast
#        forecast.start_date=${forecast_start_date} \
#        model.forecast_steps=${forecast_steps} \
#        model.base_dt=${forecast_dt} \
#        forecast.output_frequency=${output_freq} \
#        dataset.time_resolution=${forecast_data_time_resol} \
#        --config-name=paradis_settings_forecast

#logs/lightning_logs/version_rk4_8yr_${train_dt}_1step_forcing_500epoch/checkpoints/best.ckpt 

#cd results
#zip /home/siw001/hall6/paradis_model_rk4/testoutput-vicky/forecast_plots/${forecast_inf}.zip *
#zip /home/siw001/hall6/paradis_model_rk4/testoutput-vicky/forecast_plots/${forecast_inf}_data.zip */*/*
#rm -rf *









