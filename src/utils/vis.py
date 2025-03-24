import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image 
from IPython.display import Image

import time
import shutil 

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import json
from itertools import combinations, permutations

# Third-party library imports
import torch
import geoopt
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from openai import OpenAI
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D

# Lightning imports
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

def get_embeddings(model, dataloaders):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    all_embeddings = []
    all_states = []
    all_labels = []

    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    with torch.no_grad():
        for dataloader in dataloaders:
            for batch in dataloader: 
                images, states, labels, _, _, _ = batch
                images = images.to(device)
                embeddings = model.get_representation(images, states)
                all_embeddings.append(embeddings.cpu().numpy())
                all_states.extend(states)
                all_labels.extend(labels)

    all_embeddings = torch.tensor(np.concatenate(all_embeddings, axis=0))
    return all_embeddings, all_states, all_labels


def get_outputs(model, dataloaders):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    all_outputs = []
    all_states = []
    all_labels = []

    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    with torch.no_grad():
        for dataloader in dataloaders:
            for batch in dataloader: 
                images, states, labels, _, _, _ = batch
                images = images.to(device)
                outputs = model(images, states)
                all_outputs.append(outputs.cpu().numpy())
                all_states.extend(states)
                all_labels.extend(labels)

    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs, all_states, all_labels


def compute_accuracy(model, dataloader):
    all_outputs, all_states, all_labels = get_outputs(model, dataloader)

    results = {}
    for state in set(all_states):
        idxs = [i for i, s in enumerate(all_states) if s == state]
        outputs = torch.tensor(all_outputs[idxs])
        labels = torch.tensor([all_labels[i] for i in idxs])
        accuracies = model.compute_accuracy(outputs, labels)
        results[state] = accuracies
    
    all_outputs = torch.tensor(all_outputs)
    all_labels = torch.tensor(all_labels)
    results['overall'] = model.compute_accuracy(all_outputs, all_labels)
    return results


def visualize_embeddings(model, dataloader, output_path, by_pred=True, ax=None):
    plot_fn, plot_type = get_vis_function(model) 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ax = plot_fn(model, dataloader, output_path=output_path, pred_only=by_pred, ax=ax)
    return ax


def get_vis_function(model): 
    if model.args.num_hyp_layers > 0:
        plot_type = f'poincare={model.args.hyp_output_dim}'
        if model.args.hyp_output_dim == 2:
            plot_fn =  plot_poincare_disk 
        elif model.args.hyp_output_dim == 3:
            plot_fn =  plot_poincare_ball 
        else:
            plot_fn =  plot_umap 
    else:
        plot_fn =  plot_tsne
        plot_type = 'tsne'
    
    return plot_fn, plot_type


def plot_poincare_disk(model, dataloader, r=1.0, output_path=None, pred_only=False, ax=None):
    assert model.is_hyperbolic
    embeddings_poincare, labels, _ = get_embeddings(model, dataloader)
    assert embeddings_poincare.shape[1] == 2, f'embedding dims are {embeddings_poincare.shape}'
    embeddings_2d = embeddings_poincare

    if ax is None:
        plt.figure(figsize=(8, 6)) 
        ax = plt.gca()
        circle = plt.Circle((0, 0), r, color='black', fill=False)
        ax.add_artist(circle)

    if pred_only:
        label_to_pred = {label: label.split('(')[0] for label in set(labels)}
        pred_to_idx = {pred: index for index, pred in enumerate(sorted(list(set(label_to_pred.values()))))}
        label_to_idx = {string: pred_to_idx[label_to_pred[string]] for index, string in enumerate(sorted(list(set(labels))))}
        plot_labels = pred_to_idx
    else:
        label_to_idx = {string: index for index, string in enumerate(sorted(list(set(labels))))}
        plot_labels = label_to_idx

    idxs = np.array([label_to_idx[label] for label in labels]) 
        
    for label, idx in plot_labels.items():
        ax.scatter(embeddings_2d[idxs == idx, 0], embeddings_2d[idxs == idx, 1], label=label, alpha=0.5)
              
    ax.set_title("Poincaré Disk Visualization")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect('equal', adjustable='box')
        
    ax.legend(title='States', loc='center left', bbox_to_anchor=(1.0, 0.75))

    if output_path is not None:
        print(f"Saving to {output_path}")
        plt.savefig(output_path)
        plt.close()
    
    return ax
     

def plot_poincare_ball(model, dataloader, output_path=None, pred_only=False, ax=None):
    assert model.is_hyperbolic
    embeddings_poincare, labels, _ = get_embeddings(model, dataloader)
    assert embeddings_poincare.shape[1] == 3, f'embedding dims are {embeddings_poincare.shape}'
    embeddings_3d = embeddings_poincare

    if pred_only:
        label_to_pred = {label: label.split('(')[0] for label in set(labels)}
        pred_to_idx = {pred: index for index, pred in enumerate(sorted(list(set(label_to_pred.values()))))}
        label_to_idx = {string: pred_to_idx[label_to_pred[string]] for index, string in enumerate(sorted(list(set(labels))))}
        plot_labels = pred_to_idx
    else:
        label_to_idx = {string: index for index, string in enumerate(sorted(list(set(labels))))}
        plot_labels = label_to_idx

    idxs = np.array([label_to_idx[label] for label in labels]) 

    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot the unit sphere
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = np.sin(v) * np.cos(u)
    y = np.sin(v) * np.sin(u)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="lightgrey", alpha=0.5)
    
    # Plot the points inside the unit sphere
    # for label, idx in label_to_idx.items():
    for label, idx in plot_labels.items():
        ax.scatter(embeddings_3d[idxs == idx, 0], embeddings_3d[idxs == idx, 1], embeddings_3d[idxs == idx, 2], label=label, alpha=0.5)
  
    # Set the limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Embeddings on Poincaré Ball')

    ax.legend(title='States', loc='center left', bbox_to_anchor=(1.05, 0.75))
    
    if output_path is not None:
        plt.savefig(output_path)
        plt.close()

    return ax  