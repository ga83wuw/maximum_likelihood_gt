from PIL import Image
import glob
from pathlib import Path
import os

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
import time
import argparse

import json

PROB = True

def load_masks(input_path_AR, input_path_HS, input_path_SG, input_path_avrg, idx):

    input_path_AR = Path(input_path_AR)
    input_path_HS = Path(input_path_HS)
    input_path_SG = Path(input_path_SG)

    all_masks_AR = glob.glob(os.path.join(input_path_AR, '*.png'))
    all_masks_HS = glob.glob(os.path.join(input_path_HS, '*.png'))
    all_masks_SG = glob.glob(os.path.join(input_path_SG, '*.png'))
    all_masks_AR.sort()
    all_masks_HS.sort()
    all_masks_SG.sort()

    input_path_avrg = Path(input_path_avrg)
    all_masks_avrg = glob.glob(os.path.join(input_path_avrg, '*.png'))
    all_masks_avrg.sort()

    input_mask_AR = Image.open(all_masks_AR[idx]).convert("L")
    input_mask_HS = Image.open(all_masks_HS[idx]).convert("L")
    input_mask_SG = Image.open(all_masks_SG[idx]).convert("L")

    # Convert grayscale images to arrays
    array_AR = np.array(input_mask_AR)
    array_HS = np.array(input_mask_HS)
    array_SG = np.array(input_mask_SG)

    input_mask_avrg = Image.open(all_masks_avrg[idx]).convert("L")
    array_avrg = np.array(input_mask_avrg)

    return array_AR, array_HS, array_SG, array_avrg

def compute_prob_masks(array_AR, array_HS, array_SG):
    
    array_AR = array_AR
    array_HS = array_HS
    array_SG = array_SG
    
    prob_array = (array_AR + array_HS + array_SG) / 3
    
    return prob_array

def compute_alpha_product(A, masks, w, h):
    
    M = len(A)
    
    products = [1, 1]
    
    for j in range(len(products)):
        for i in range(M):
            
            y = int(masks[i][int(w), int(h)])
            products[j] = products[j] * A[i][j, y]
        
    return products[0], products[1]

def compute_density(masks):
    
    no = len(masks)
    
    rho_0 = 0
    rho_1 = 0
    
    for i in range(no):
        rho_0 = rho_0 + np.count_nonzero(masks[i] == 0)
        rho_1 = rho_1 + np.count_nonzero(masks[i] == 1)
        
    rho_0 /= no
    rho_1 /= no
    
    s_rho = rho_0 + rho_1
    
    rho_0 /= s_rho
    rho_1 /= s_rho
    
    return rho_0, rho_1

def compute_prob(A, masks, w, h, rho_0 = 1., rho_1 = 1.):
    
    e = 1e-5
    
    alpha_0, alpha_1 = compute_alpha_product(A, masks, w, h)
    
    norm = alpha_0 * rho_0 + alpha_1 * rho_1 + e
    
    p0 = (alpha_0 * rho_0 + e) / norm
    p1 = (alpha_1 * rho_1 + e) / norm
    
    return p0, p1

def compute_new_gt(A, masks, compute_density = lambda masks: (1, 1)):
    
    W, H = masks[0].shape[0], masks[0].shape[1]
    
    new_mask = np.zeros((W, H), np.float16)
    
    rho_0, rho_1 = compute_density(masks)
    
    # Create a progress bar using tqdm
    progress_bar = tqdm(total=W*H, desc='Computing', unit='pixel')
    for w in range(W):
        for h in range(H):
            
            p0, p1 = compute_prob(A, masks, w, h, rho_0, rho_1)

            if PROB:
                
                new_mask[w, h] = (1. - p0) if p0 > p1 else p1

            else:
                
                new_mask[w, h] = 0 if p0 > p1 else 1
            # Update the progress bar
            progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()
    
    return new_mask
            
def plot_comparison(array_prob, new_mask_uint8, save_path, idx):

    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Display the images in the subplots
    axs[0].imshow(array_prob)
    axs[0].set_title("prob Mask")
    axs[0].axis("off")

    axs[1].imshow(new_mask_uint8)
    axs[1].set_title("new Mask")
    axs[1].axis("off")

    axs[2].imshow((new_mask_uint8 / 255. - array_prob) * 255.)
    axs[2].set_title("diff Mask")
    axs[2].axis("off")

    # Adjust the spacing between subplots
    plt.tight_layout()

    os.makedirs(save_path, exist_ok = True)

    # Save the figure to a file
    save_file = os.path.join(save_path, f"comparison_plot_{idx + 1}.png")
    plt.savefig(save_file)

    # Close the figure
    plt.close(fig)

def load_confusion_matrices(confusion_matrix_paths):
    
    matrices = []
    for path in confusion_matrix_paths:
        matrix = np.loadtxt(path)
        matrices.append(matrix)
    
    return matrices

if __name__ == '__main__':
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description = "Maximum Likelihood Ground Truth Generation")
    parser.add_argument("config_file", type = str, help = "Path to the JSON configuration file.")
    args = parser.parse_args()

    # Read the config_mlgt.json file
    with open(args.config_file) as f:
        config = json.load(f)

    # Extract the required values from the config
    input_path_AR = config['input_path_AR']
    input_path_HS = config['input_path_HS']
    input_path_SG = config['input_path_SG']
    input_path_avrg = config['input_path_avrg']
    idx = config['idx']
    cm_AR = config['cm_AR_path']
    cm_HS = config['cm_HS_path']
    cm_SG = config['cm_SG_path']
    
    # Retrieve the save path from the config
    save_path = config["save_path"]

    confusion_matrix_paths = [cm_AR, cm_HS, cm_SG]

    # Load the masks
    array_AR, array_HS, array_SG, array_avrg = load_masks(input_path_AR, input_path_HS, input_path_SG, input_path_avrg, idx)

    # Compute the probability masks
    prob_array = compute_prob_masks(array_AR, array_HS, array_SG)

    # Load CMs
    A = load_confusion_matrices(confusion_matrix_paths)

    # Compute the new ground truth mask
    masks = [array_AR / 255., array_HS / 255., array_SG / 255.]
    new_mask = compute_new_gt(A, masks, compute_density)

    # Convert the new mask to uint8 for visualization
    new_mask_uint8 = (new_mask * 255).astype(np.uint8)

    # Plot the comparison
    plot_comparison(prob_array, new_mask_uint8, save_path, idx)