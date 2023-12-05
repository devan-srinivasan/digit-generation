"""
This file has a suite of interpretability techniques used to assess the diffusion model.
It's powered by the PyTorch interpretability library - Captum
In this file, one can run experiments with Grad-CAM, Saliency, and Integrated Gradients
"""
import torch, matplotlib.pyplot as plt
import numpy as np
import os
from torch import tensor
from models import MyUNet, MyDDPM
from argparse import ArgumentParser
from captum.attr import LayerGradCam, IntegratedGradients, Saliency

# settings
n_steps = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
store_path = "ddpm_mnist.pt"
k = 3

# load trained model
model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
model.load_state_dict(torch.load(store_path, map_location=device))

# turn off non-training params.
model.eval()

def run_method(model, layer, target, method, input_noise, t=False):
    """
    model is the model to analyze
    layer is a layer within such model to assess contributions (traditionally last layer of network)
    target is the target classes -- here we conventionally assess gradient contributions for each pixel, indexed [0 - 784]
    t is time step in denoising process U-Net is estimating, False is not applicable (or unspecified)
    method is one of grad-cam, saliency, integrated-gradients

    returns the attributions given specified method
    """
    options = {
        'grad-cam': LayerGradCam,
        'saliency': Saliency,
        'integrated-gradients': IntegratedGradients,
    }
    layer_attr = options[method](model, layer) if method == 'grad-cam' else options[method](model)
    attr = layer_attr.attribute(input_noise, target=target, additional_forward_args=tensor([t]) if t else None)
    return attr

def run_method_whole(method, noise, time):
    """
    This runs method for each pixel, summing up the values for 
    a given noise image and time step for our model to denoise
    """
    summed_attributions = torch.zeros(1, 1, 28, 28)
    for k in range(784):
        attr = run_method(model=model.network, layer=model.network.up1[2], target=k, t=time, input_noise=noise, method=method)
        summed_attributions += attr
    contrast_scale = 1
    return summed_attributions * contrast_scale

def overlay(noise_img, layers, labels = [], no_overlay=True):
    """
    Overlays all images in layer (tensors) on image tensor
    """
    import matplotlib.pyplot as plt
    n_cols = len(layers)
    fig, axes = plt.subplots(1, n_cols, figsize=(10, 8))
    for c in range(n_cols):
        axes[c].set_xticks([])
        axes[c].set_yticks([])
        if not no_overlay: axes[c].imshow(noise_img, 'gray', interpolation='none')
        axes[c].imshow(layers[c], 'Oranges', interpolation='none', alpha=0.9)
        if len(labels) == len(layers): axes[c].set_title(f't={labels[c]}') 
    plt.show()

def plot_images(
    image_tensors: tensor
):
    """
    Given a tensor of image tensors, plot them in a grid
    """
    n_images = len(image_tensors)

    # Create a figure and axis objects
    if 1 < n_images <= 10:
        fig, axes = plt.subplots(1, n_images, figsize=(10, 8))

        # Loop through each tensor and plot it on the grid
        for i, tensor in enumerate(image_tensors):
            ax = axes[i]
            ax.imshow(tensor.detach().cpu().numpy(), cmap='gray')  # Use 'cmap' for grayscale images
            ax.axis('off')  # Turn off axis labels
    elif n_images == 1:
        plt.imshow(image_tensors[0].detach().cpu().numpy(), cmap='gray')
    else:
        # set up grid
        n_cols = int(n_images ** 0.5)
        n_rows = n_images // n_cols + int(n_images % n_cols > 0)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))

        # Loop through each tensor and plot it on the grid
        for i, tensor in enumerate(image_tensors):
            ax = axes[i // n_cols, i % n_cols]
            ax.imshow(tensor.detach().cpu().numpy(), cmap='gray')  # Use 'cmap' for grayscale images
            ax.axis('off')  # Turn off axis labels

    # Adjust layout and display the grid of images
    plt.tight_layout()
    plt.show()

def constant_noise_exp(method, timesteps):
    """
    Using a constant noise image, we can see at different time steps
    how are model operates under different interpretability methods.
    i.e. for grad cam, we can see at different timesteps what localized regions our model 
    pays the most attention to throughout the denoising process.
    """
    noise = torch.randn(1, 1, 28, 28, requires_grad=True)
    results = [
        run_method_whole(method, noise, t)[0][0].detach().numpy() for t in timesteps
    ]
    overlay(noise[0][0].detach().numpy(), results, [str(e) for e in timesteps])

def live_noise_exp(method, timesteps):
    """
    Unlike the constant noise step, we can also monitor the model throughout
    the actual denoising process interrupting it to assess it's attributions
    with the given method. That is what this experiment does.

    NOTE timesteps must be ordered descending!
    """
    # helpful variables
    N = 1

    # Starting from random noise
    x = torch.randn(N, 1, 28, 28).to(device)

    if method != 'all':
        results = []
    else:
        results1 = []
        results2 = []
        results3 = []
    next_idx = 0

    for idx, t in enumerate(list(range(model.n_steps))[::-1]):
        # Estimating noise to be removed
        time_tensor = (torch.ones(N, 1) * t).to(device).long()
        eta_theta = model.backward(x, time_tensor).reshape(-1, 1, 28, 28)    # TODO backward

        alpha_t = model.alphas[t]
        alpha_t_bar = model.alpha_bars[t]

        # calculate interpretability attribution matrix
        if next_idx < len(timesteps) and t == timesteps[next_idx]:
            if method != 'all':
                results.append(run_method_whole(method, x, t)[0][0].detach().numpy())
            else:
                results1.append(run_method_whole('grad-cam', x, t)[0][0].detach().numpy())
                results2.append(run_method_whole('saliency', x, t)[0][0].detach().numpy())
                results3.append(run_method_whole('integrated-gradients', x, t)[0][0].detach().numpy())
            next_idx += 1

        # Partially denoising the image
        x = (1 / alpha_t.sqrt()) * (
            x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
        )
        
        if t > 0:
            z = torch.randn(N, 1, 28, 28).to(device)

            # Option 1: sigma_t squared = beta_t
            beta_t = model.betas[t]
            sigma_t = beta_t.sqrt()

            # Option 2: sigma_t squared = beta_tilda_t
            # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
            # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
            # sigma_t = beta_tilda_t.sqrt()

            # Adding some more noise like in Langevin Dynamics fashion
            x = x + sigma_t * z
    labels = [str(e) for e in timesteps]
    if method != 'all':
        overlay(x[0][0].detach().numpy(), results, labels)
    else:
        overlay(x[0][0].detach().numpy(), results1, labels)
        overlay(x[0][0].detach().numpy(), results2, labels)
        overlay(x[0][0].detach().numpy(), results3, labels)
    plot_images(x.squeeze(1))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument( "--method", type=str, default="saliency", help="One of 'grad-cam', 'saliency', 'integrated-gradients', 'all'. Note 'all' must be run with type 'live'")
    parser.add_argument("--type", type=str, default="live", help="One of 'constant', 'live'")
    parser.add_argument("--timesteps", type=str, default="900, 500, 100, 20, 10, 5, 3, 1", help="timesteps in DESCENDING ORDER, comma separated")
    args = vars(parser.parse_args())
    args['timesteps'] = [int(x) for x in args['timesteps'].split(',')]
    print(f'Running Captum Experiments \n\t method:{args["method"]} \n\t type:{args["type"]} \n\t timesteps:{args["timesteps"]}')
    if args['type'] == 'constant':
        constant_noise_exp(args['method'], args['timesteps'])
    else:
        live_noise_exp(args['method'], args['timesteps'])