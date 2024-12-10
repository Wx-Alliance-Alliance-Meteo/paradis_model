# PARADIS: **P**hysically-informed **A**dvection, **R**eaction **A**nd **DI**ffusion on the **S**phere

## Key Features

- **Physically-informed neural architecture:** Combines convolutional neural networks with operators akin to the **A**dvection, **R**eaction, and **D**iffusion (ADR) equation to handle complex physical phenomena.
- **Spherical Semi-Lagrangian advection operator:** Enables the non-local transformation of information in spherical coordinates
- **Geocyclic Padding:** Addresses the challenge of representing the Earth’s spherical geometry in convolutional neural networks

#### Dependencies
Necessary python packages can be installed using **pip3**:

```
pip install -r requirements.txt --break-system-packages
```

#### Usage
##### Training
For training, the `train.yaml` file in the `config/` directory is used to manage inputs
```
python train.py [override_args]
```
where `[override_args]` can override the inputs in the config file.


#### Dataset
Currently, the 4-degree, 13 layer dataset is located at `/home/cap003/hall5/gatmosphere_data/ERA5_4deg_unified`

Alternatively, you can download a subset of the data using
```
wget https://hpfx.collab.science.gc.ca/~mmo000/ERA5_4deg_2020_2022_unified.tgz
```

#### Acknowledgements

This project draws significant inspiration from the paper, ["Advection Augmented Convolutional Neural Networks"](https://arxiv.org/abs/2406.19253) from Niloufar Zakariaei, Siddharth Rout, Eldad Haber, Moshe Eliasof
