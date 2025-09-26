**PARADIS**: **P**hysically inspired **A**dvection, **R**eaction **A**nd **DI**ffusion on the **S**phere

#### Key Features

- **Physically-informed neural architecture:** Combines convolutional neural networks with operators akin to the advection, diffusion, and reaction (ADR) equation to handle complex physical phenomena.
- **Spherical Semi-Lagrangian advection operator:** Enables the non-local transformation of information in spherical coordinates.
- **Geocyclic Padding:** Addresses the challenge of representing the Earth’s spherical geometry

#### Dependencies
Necessary python packages can be installed using **pip**:

```
pip install -r requirements.txt --break-system-packages
```

#### Usage
##### Training
For training, the `paradis_settings.yaml` file in the `config/` directory is used to manage inputs
```
python train.py [override_args]
```
where `[override_args]` can override the inputs in the config file.


#### Dataset

Download the original dataset from WeatherBench 2:

```
cd scripts
bash download_dataset.sh OUTPUT_DIR
```
where OUTPUT_DIR is the destination directory and then preprocess it

```
python scripts/preprocess_weatherbench_data.py -i /path/to/ERA5/5.625deg_wb2 -o /path/to/ERA5/5.65deg
```

#### Acknowledgements

This project draws significant inspiration from the paper ["Advection Augmented Convolutional Neural Networks"](https://arxiv.org/abs/2406.19253) by Niloufar Zakariaei, Siddharth Rout, Eldad Haber and Moshe Eliasof.

The Geocyclic Padding method is inspired from the paper ["Karina: An efficient deep learning model for global weather forecast"](https://arxiv.org/abs/2403.10555) by
Minjong Cheon, Yo-Hwan Choi, Seon-Yu Kang, Yumi Choi, Jeong-Gil Lee and Daehyun Kang.

#### Attention version 1

This version will include an input projection layer that will use Attention mecanisme to provide global information to the advective layer. 
However other version will explore alternative way to incorporate attention, namely in the advective network, the diffusion-reaction network 
and the decoding layer.