# Denoising-Milky-Way-with-NF

## Overview
This repository was used to train models and generate results presented in Section 3.3 Fiducial setup of the Denoising Milky Way stellar survey data with normalizing flow models. Which contains the implementation of a novel machine-learning approach using Normalizing Flows for denoising and density estimation of stellar survey data, particularly from the Gaia dataset. The method is designed to handle heteroscedastic noise in the data, enabling precise characterization of intricate Milky Way substructures such as moving groups and the phase spiral.



## Usage
1. **Data Preprocessing**:
   - Use `src/utils.py` to preprocess Gaia data and generate mock datasets.
2. **Model Training**:
   - Train Normalizing Flow models using `src/flow.py`. Customize hyperparameters such as the number of transformations, hidden layers, and batch size.
3. **Visualization**:
   - Use `src/plotting.py` to visualize velocity distributions, phase-space spirals, and other substructures.
4. **Example Notebook**:
   - Refer to `example/example.ipynb` for a demo that demonstrating the core concept.
5. **Trained model**:
   - Check `model` for the trained models and associated plots that used in the paper.

## Citation
If you use this code in your research, please cite the associated paper:
```
[Paper citation details here]
```

## Acknowledgments
This project uses libraries such as `torch`, `zuko`, `astropy`, and `matplotlib` for machine learning, astrophysical computations, and visualization.