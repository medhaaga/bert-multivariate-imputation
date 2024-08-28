import os
import torch
from src.os_utils import (get_results_path)

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams["axes.labelsize"] = 15
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.serif'] = ['Times New Roman']


def plot_missing_acc_data(test, mask):

    indices_missing1 = torch.nonzero(mask[:,0]).flatten()


    # # Mask out the values equal to 100 for plotting
    data_plot = test.clone()
    data_plot[mask] = float('nan')

    # # Create a time axis
    time = torch.arange(len(test))

    # # Plot the time series data
    plt.figure(figsize=(12, 4))
    cmap = plt.get_cmap('Set2') 
    temporal_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    for i in range(data_plot.shape[1]):
        plt.plot(time, data_plot[:,i], marker='o', markersize=1, linewidth=2, color=cmap(i/data_plot.shape[1]), label = temporal_cols[i])

    # Add vertical lines at the indices where data is 100
    for idx in indices_missing1:
        plt.axvline(x=idx.item(), color='grey', linewidth=2, alpha=0.1)

    plt.xlabel('Time $[h]$')
    plt.ylabel(r'Concentration $[\mu g/m^3]$')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=data_plot.shape[1])
    plt.tight_layout()
    plt.savefig(os.path.join(get_results_path(), 'missing_test.png'))
    plt.show()

def plot_imputed_acc_data(test, mask, static, model, device, mask_token):

    masked_test = test.clone()
    masked_test[mask] = mask_token

    indices_missing1 = torch.nonzero(mask[:,0]).flatten()

    with torch.no_grad():
        predictions = model(masked_test.unsqueeze(0).to(device), static.unsqueeze(0).to(device))
        
    mask_pred = predictions['mask_predictions'].squeeze(0).detach().cpu()
    all_pred = predictions['predictions'].squeeze(0).detach().cpu()


    masked_test[mask] = mask_pred

    # Create a time axis
    time = torch.arange(len(test))

    # Plot the time series data
    plt.figure(figsize=(12, 4))
    cmap = plt.get_cmap('Set2') 
    temporal_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    for i in range(masked_test.shape[1]):
        if i==0:
            plt.plot(time, masked_test[:,i], marker='o', markersize=1, linewidth=1.5, color='pink', label='Imputed')
            plt.plot(time, test[:,i],  marker='o', markersize=1, linewidth=1, color=cmap(i/masked_test.shape[1]), label=temporal_cols[i])
        else:
            plt.plot(time, masked_test[:,i], marker='o', markersize=1, linewidth=1.5, color='pink')
            plt.plot(time, test[:,i],  marker='o', markersize=1, linewidth=1, color=cmap(i/masked_test.shape[1]), label=temporal_cols[i])
    
    # Add vertical lines at the indices where data is 100

    for idx in indices_missing1:
        plt.axvline(x=idx.item(), color='grey', linewidth=1.5, alpha=0.1)

    plt.xlabel(r'Time $[h]$')
    plt.ylabel(r'Concentration $[\mu g/ m^3]$')

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=7)
    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(get_results_path(), 'imputed_test.png'))
    return all_pred