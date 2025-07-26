import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import os
import pickle
import time

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x, num_frequencies=6, incl_input=True):

    """
    Apply positional encoding to the input.

    Args:
    x (torch.Tensor): Input tensor to be positionally encoded.
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding (default: 6).
    incl_input (optional, bool): If True, concatenate the input with the
        computed positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor. shape: [N, D*(2*num_frequencies+1)] or [N, D*2*num_frequencies] if incl_input is False.
    """

    results = []
    if incl_input:
        results.append(x)
    #############################  TODO 1(a) BEGIN  ############################
    N, D = x.shape
    for i in range(num_frequencies):
        freq = (2 ** i) * torch.pi
        for fn in [torch.sin, torch.cos]:
            encoded = fn(x * freq) 
            results.append(encoded)
    #############################  TODO 1(a) END  ##############################
    return torch.cat(results, dim=-1)

def get_rays(height, width, intrinsics, w_R_c, w_T_c):
    """
    Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).

    Args:
    height: the height of an image.
    width: the width of an image.
    intrinsics: camera intrinsics matrix of shape (3, 3).
    w_R_c: Rotation matrix of shape (3,3) from camera to world coordinates.
    w_T_c: Translation vector of shape (3,1) that transforms

    Returns:
    ray_origins (torch.Tensor): A tensor of shape (height, width, 3) denoting the centers of
      each ray. Note that despite that all rays share the same origin, here we ask you to return
      the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape (height, width, 3) denoting the
      direction of each ray.
    """

    device = intrinsics.device
    ray_directions = torch.zeros((height, width, 3), device=device)
    ray_origins = torch.zeros((height, width, 3), device=device)
    u, v = torch.meshgrid(
        torch.arange(0, width, device=device),
        torch.arange(0, height, device=device),
        indexing='xy'  
    )
    x_normalized = (u - intrinsics[0, 2]) / intrinsics[0, 0]
    y_normalized = (v - intrinsics[1, 2]) / intrinsics[1, 1]
    cam_dirs = torch.stack([x_normalized, y_normalized, torch.ones_like(x_normalized)], dim=-1)  
    ray_origins[:] = w_T_c.view(1, 1, 3).expand(height, width, 3)  
    ray_directions = torch.matmul(w_R_c, cam_dirs.reshape(-1, 3).T).T.reshape(height, width, 3) 

    return ray_origins, ray_directions

def stratified_sampling(ray_origins, ray_directions, near, far, samples):

    """
    Sample 3D points on the given rays. The near and far variables indicate the bounds of sampling range.

    Args:
    ray_origins: Origin of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    ray_directions: Direction of each ray in the "bundle" as returned by the
      get_rays() function. Shape: (height, width, 3).
    near: The 'near' extent of the bounding volume.
    far:  The 'far' extent of the bounding volume.
    samples: Number of samples to be drawn along each ray.

    Returns:
    ray_points: Query 3D points along each ray. Shape: (height, width, samples, 3).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).
    """

    #############################  TODO 2.2 BEGIN  ############################
    t_vals = torch.linspace(0.0, 1.0, steps=samples, device=ray_origins.device)
    depth_points = near + (far - near) * t_vals  

    H, W, _ = ray_origins.shape
    depth_points = depth_points.view(1, 1, samples).expand(H, W, samples)  
    ray_directions = ray_directions.unsqueeze(2)  
    ray_origins = ray_origins.unsqueeze(2)      

    ray_points = ray_origins + depth_points.unsqueeze(-1) * ray_directions  


    #############################  TODO 2.2 END  ############################
    return ray_points, depth_points


class nerf_model(nn.Module):

    """
    Define a NeRF model comprising eight fully connected layers and following the
    architecture described in the NeRF paper.
    """

    def __init__(self, filter_size=256, num_x_frequencies=10, num_d_frequencies=4):
        super().__init__()

        input_pos_dim = (num_x_frequencies * 2 + 1) * 3
        input_dir_dim = (num_d_frequencies * 2 + 1) * 3

        #############################  TODO 2.3 BEGIN  ############################
        self.layers = nn.ModuleDict({
            'layer_1': nn.Linear(input_pos_dim, filter_size),
            'layer_2': nn.Linear(filter_size, filter_size),
            'layer_3': nn.Linear(filter_size, filter_size),
            'layer_4': nn.Linear(filter_size, filter_size),
            'layer_5': nn.Linear(filter_size, filter_size),  # skip connection
            'layer_6': nn.Linear(filter_size+ input_pos_dim, filter_size),
            'layer_7': nn.Linear(filter_size, filter_size),
            'layer_8': nn.Linear(filter_size, filter_size),
            'layer_9': nn.Linear(filter_size, 1),
            'layer_10': nn.Linear(filter_size, filter_size),
            'layer_11': nn.Linear(filter_size + input_dir_dim, 128),
            'layer_12': nn.Linear(128, 3)
        })

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #############################  TODO 2.3 END  ############################

    def forward(self, x, d):
        #############################  TODO 2.3 BEGIN  ############################
        x_input = x

        x = self.relu(self.layers['layer_1'](x))
        x = self.relu(self.layers['layer_2'](x))
        x = self.relu(self.layers['layer_3'](x))
        x = self.relu(self.layers['layer_4'](x))
        x = self.relu(self.layers['layer_5'](x))
        x = torch.cat([x, x_input], dim=-1)
        x = self.relu(self.layers['layer_6'](x))
        x = self.relu(self.layers['layer_7'](x))
        x = self.relu(self.layers['layer_8'](x))
        sigma = self.relu(self.layers['layer_9'](x))
        features = self.layers['layer_10'](x)
        combined = torch.cat([features, d], dim=-1)
        x = self.relu(self.layers['layer_11'](combined))
        rgb = self.sigmoid(self.layers['layer_12'](x))
        #############################  TODO 2.3 END  ############################
        return rgb, sigma

def get_batches(ray_points, ray_directions, num_x_frequencies, num_d_frequencies):

    def get_chunks(inputs, chunksize = 2**15):
        """
        This fuction gets an array/list as input and returns a list of chunks of the initial array/list
        """
        return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    """
    This function returns chunks of the ray points and directions to avoid memory errors with the
    neural network. It also applies positional encoding to the input points and directions before
    dividing them into chunks, as well as normalizing and populating the directions.
    Return Shape:   ray_points_batches: List of chunks of ray points. Each chunk has shape (H, W, S, 3*(2*num_x_frequencies+1)).
                    ray_directions_batches: List of chunks of ray directions. Each chunk has shape (H, W, S, 3*(2*num_x_frequencies+1)).
    """
    #############################  TODO 2.3 BEGIN  ############################
    # Apply positional encoding to the ray points and directions (you may nomalize the directions here)
    # repeat the directions to match the dimension and number of points S
    H, W, S, _ = ray_points.shape  
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    ray_directions = ray_directions.unsqueeze(2).expand(-1, -1, S, -1)
    ray_points = ray_points.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    ray_points = positional_encoding(ray_points, num_frequencies=num_x_frequencies, incl_input=True)
    ray_directions = positional_encoding(ray_directions, num_frequencies=num_d_frequencies, incl_input=True)
    ray_points_batches = get_chunks(ray_points)
    ray_directions_batches = get_chunks(ray_directions)

    #############################  TODO 2.3 END  ############################

    return ray_points_batches, ray_directions_batches
    
def volumetric_rendering(rgb, s, depth_points):

    """
    Differentiably renders a radiance field, given the origin of each ray in the
    "bundle", and the sampled depth values along them.

    Args:
    rgb: RGB color at each query location (X, Y, Z). Shape: (height, width, samples, 3).
    s: Volume density sigma at each query location (X, Y, Z). Shape: (height, width, samples).
    depth_points: Sampled depth values along each ray. Shape: (height, width, samples).

    Returns:
    rec_image: The reconstructed image after applying the volumetric rendering to every pixel.
    Shape: (height, width, 3)
    """

    #############################  TODO 2.4 BEGIN  ############################
    device = rgb.device
    H, W, N = s.shape
    s = nn.functional.relu(s)
    delta = torch.zeros((H, W, N), device=device)
    delta[:, :, :-1] = depth_points[:, :, 1:] - depth_points[:, :, :-1]
    delta[:, :, -1] = 1e9 

    alpha = 1.0 - torch.exp(-s * delta)  
    weights = (torch.cumprod(torch.cat([torch.ones((H, W, 1), device=device), 
                                        1.0 - alpha + 1e-10], dim=-1), dim=-1))[:, :, :-1]  * alpha  
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=2) 
    rec_image = rgb_map.clamp(0, 1)


    #############################  TODO 2.4 END  ############################

    return rec_image


def one_forward_pass(height, width, intrinsics, pose, near, far, samples, model, num_x_frequencies, num_d_frequencies):

    #############################  TODO 2.5 BEGIN  ############################
    row0 = pose[0]
    row1 = pose[1]
    row2 = pose[2]

    w_R_c = torch.stack([
        row0[0:3],
        row1[0:3],
        row2[0:3]
    ], dim=0)

    w_T_c = torch.stack([
        row0[3].unsqueeze(0),
        row1[3].unsqueeze(0),
        row2[3].unsqueeze(0)
    ], dim=0)
    ray_origins, ray_directions = get_rays(height, width, intrinsics, w_R_c, w_T_c)

    ray_points, depth_points = stratified_sampling(ray_origins, ray_directions, near, far, samples)

    ray_points_batches, ray_directions_batches = get_batches(
        ray_points, ray_directions, num_x_frequencies, num_d_frequencies
    )

    rgb = None
    sigma = None
    for i in range(len(ray_points_batches)):
        rgb_batch, sigma_batch = model(ray_points_batches[i], ray_directions_batches[i])
        if rgb is None:
            rgb = rgb_batch
            sigma = sigma_batch
        else:
            rgb = torch.cat((rgb, rgb_batch), dim=0)
            sigma = torch.cat((sigma, sigma_batch), dim=0)

    rgb = rgb.reshape(height, width, samples, 3)
    sigma = sigma.reshape(height, width, samples)
    rec_image = volumetric_rendering(rgb, sigma, depth_points)


    #############################  TODO 2.5 END  ############################

    return rec_image