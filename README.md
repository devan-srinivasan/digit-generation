# Generating Handwritten Digits
Project for CSC311 Fall 2023 at the University of Toronto
## Overview
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
Now you can run the model and experiments. 
### Hyperparameter Testing
### Principal Component Analysis
Run `python --k 3 --n 100 --c 3 --gif --outdir visualizations pca_visualizations.py`  <br></br>The script will save the according generated images and PCA analysis, as well as "eigenimages" (each visualized principal eigenvector) to the provided `outdir` folder (created if argument not specified). Note the arguments `k, n, c`respectively represent the number of principal components, number of image samples, and cutoff for which states are analyzed (default values as stated in the command above). The gif flag will generate a fun PCA gif across all denoising iterations.
### CNN Gradient Analysis
### CNN Kernel Visualizations


<h2>Sources</h2>
