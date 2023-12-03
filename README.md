# Generating Handwritten Digits
Project for CSC311 Fall 2023 at the University of Toronto
## Overview
This project aims to assess the interpretability of diffusion models for image generation, with the chosen scope to be generating handwritten digits. We do so by running experiments on both the U-Net noise predictor and denoising process, allowing us to gain insight into what guides the generative process apart from the known mathematics and programmatic strategy. The project, including its results, can be found here [ add link if possible ]
## Usage
We used virtual environments for running this project. Alternatively you can install the requirements from `requirements.txt` on your machine. For the virtual env., make a `venv` folder and start a virtual environment in 
```
python -m venv venv
```
Then activate and install the dependencies with 
```
source venv/bin/activate
pip install -r requirements.txt
```
To deactivate the virtual environment just enter `deactivate` <br>
Now you can run the model and experiments.  Note that all experiments, when ran generate one (or many) visualizations.
### Hyperparameter Testing
### Principal Component Analysis
`python pca.py` will run the experiment with default settings. Alternatively, you can specify them as follows below:
- `--k` is the number of principal components
- `--n` is the number of samples
-  `--timesteps` is a comma separated list of integers that represent time steps we execute PCA at.

### Captum
`python captum_exp.py` will run the experiment with default settings (default is grad-cam). Alternatively, you can specify them as follows below:
- `--method` is one of `grad-cam`, `saliency`, `integrated-gradients`, or `all`
- `--type` is the type of noise input to the analysis, constant for all timesteps, or live during denoising (usually more interesting)
-  `--timesteps` is a comma separated list of integers that represent time steps we execute PCA at.

**Grad-CAM:** This will generate overlayed maps that signal the relevance of the last convolution layer to the final output, via it's gradients. One can interpret what regions of the segmented image (from U-Net) are then used for the prediction decision.
**Saliency:** This will generate overlayed maps that signal the attribution of the input features themselves to the final output.  One can interpret what pixels of the original noisy image are then used for the prediction decision in U-Net.
**Integrated Gradients:** [ TODO ]
### Kernel Visualizations
[ yet to be implemented ]
### Conditioned Sampling Saliency
[ yet to be implemented ]


<h2>Sources</h2>
