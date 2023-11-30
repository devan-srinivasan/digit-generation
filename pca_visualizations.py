"""
Run PCA on different states of the DDPM model
"""
import torch, matplotlib.pyplot as plt
import numpy as np
import einops
import imageio
from torch import tensor
from models import MyUNet, MyBlock, MyDDPM

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
    image_tensors: tensor,
    title: str = "Image"
):
    """
    Given a tensor of image tensors, plot them in a grid
    """
    n_images = len(image_tensors)
    n_rows = int(np.sqrt(n_images))
    n_cols = int(np.ceil(n_images / n_rows))

    # Create a figure and axis objects
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))

    # Loop through each tensor and plot it on the grid
    for i, tensor in enumerate(image_tensors):
        ax = axes[i // n_cols, i % n_cols]
        ax.imshow(tensor.detach().cpu().numpy(), cmap='gray')  # Use 'cmap' for grayscale images
        ax.axis('off')  # Turn off axis labels
        ax.set_title(f'{title} {i+1}')  # Set a title for each image if needed

    # Adjust layout and display the grid of images
    plt.tight_layout()
    plt.show()

def generate_PCA_images(
    ddpm,
    n_samples=100,
    device=None,
    frames_per_gif=100,
    gif_name="sampling.gif",
    c=1,
    h=28,
    w=28,
    k=3
):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []
    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # helpful variables
        N = n_samples

        # Starting from random noise
        x = torch.randn(N, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(N, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )

            proj, recon = PCA(x.squeeze(1).reshape(N, -1), k)
            recon = recon.unsqueeze(1).reshape(N, 1, h, w)
            # reshape reconstructed projections into 28x28 images

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = recon.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(
                    normalized,
                    "(b1 b2) c h w -> (b1 h) (b2 w) c",
                    b1=int(n_samples**0.5),
                )
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)
            
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
    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            # Showing the last frame for a longer time
            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)
    return x

res = generate_PCA_images(ddpm=model, 
                          gif_name='latent_pca.gif',
                          k=5
                          )
plot_images(res.squeeze(1))