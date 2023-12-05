"""
This file runs PCA on N sample images at various time steps, as means to assess
what principal components are relative to the denoising process.
"""
import torch, matplotlib.pyplot as plt
import numpy as np
import einops
import imageio
import os
from torch import tensor
from models import MyUNet, MyDDPM
from argparse import ArgumentParser

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

def PCA(
        data: tensor, 
        k: int
):
    """
    executes PCA on data tensor with k principal components
    """
    # center the data
    data_mean = torch.mean(data, dim=0)
    centered_data = data - data_mean

    # covariance
    covariance_matrix = torch.matmul(centered_data.t(), centered_data) / (len(centered_data) - 1)

    # calculate eigenvectors and sort by eigenvalue
    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
    sorted_indices = torch.argsort(eigenvalues.real, descending=True)
    sorted_eigenvectors = eigenvectors[:, sorted_indices].real

    projection_matrix = sorted_eigenvectors[:, :k]

    projected_data = torch.matmul(centered_data, projection_matrix)
    reconstructed_data = torch.matmul(projected_data, projection_matrix.t()) + data_mean

    return projected_data, reconstructed_data

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

def generate_PCA_images(
    ddpm,
    n_samples=100,
    device=None,
    timesteps=[999,500,100,1],
    c=1,
    h=28,
    w=28,
    k=3
):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    # GIF STUFF
    # frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    # frames = []
    results = []
    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # helpful variables
        N = n_samples

        # Starting from random noise
        x = torch.randn(N, c, h, w).to(device)
        next_idx = 0
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(N, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor).reshape(-1, 1, 28, 28)    # TODO backward

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )
            
            if next_idx < len(timesteps) and t == timesteps[next_idx]:
                proj, recon = PCA(x.squeeze(1).reshape(N, -1), k)
                recon = recon.unsqueeze(1).reshape(N, 1, h, w)
                results.append(recon)
                next_idx += 1
            
            if t > 0:
                z = torch.randn(N, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z
    return x, results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--k", type=int, default=3, help="Number of principal components")
    parser.add_argument("--n", type=int, default=15, help="Number of image samples to generate")
    parser.add_argument("--timesteps", type=str, default="999,500,100,1", help="timesteps in DESCENDING ORDER, comma separated")
    args = vars(parser.parse_args())
    args['timesteps'] = [int(x) for x in args['timesteps'].split(',')]
    print(f"Running PCA analysis with \n\tn:{args['n']}\n\tk:{args['k']}\n\tk:{args['timesteps']}")
    denoised_img, res = generate_PCA_images(ddpm=model, 
                            n_samples=args["n"],
                            timesteps=args["timesteps"],
                            k=args["k"]
                            )
    for pca_res in res:
        plot_images(pca_res.squeeze(1))
    plot_images(denoised_img)