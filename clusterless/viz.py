import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=15)         
plt.rc('axes', titlesize=15)     
plt.rc('axes', labelsize=15) 
plt.rc('axes', linewidth=1.5)
plt.rc('xtick', labelsize=15)   
plt.rc('ytick', labelsize=15)   
plt.rc('legend', fontsize=15)   
plt.rc('figure', titlesize=15)


def plot_decoder_input(
    data,
    data_type,
    n_spikes,
    display_seconds=True,
    save_fig=False,
    out_path=None
):
    plt.figure(figsize=(6.5,5))
    plt.imshow(data.mean(0), aspect='auto', cmap='cubehelix')
    if display_seconds:
        plt.axvline(x=10, color='orange', linewidth=3, linestyle='--')
        plt.xticks([0, 10, 20, 29], ['0', '0.5', '1', '1.5'])
        plt.xlabel('time (sec)')
    else:
        plt.xlabel('time bin')
        
    cbar = plt.colorbar()
    if data_type != 'ADVI + GMM':
        cbar.ax.set_ylabel('spike count')
    else:
        cbar.ax.set_ylabel('weight')
    plt.title(f'{data_type} ({n_spikes} spikes)')
    
    if data_type == 'thresholded':
        plt.ylabel(f'{data.shape[1]} channels')
    elif data_type in ['all ks units', 'good ks units']:
        plt.ylabel(f'{data.shape[1]} units')
    elif data_type == 'ADVI + GMM':
        plt.ylabel(f'{data.shape[1]} components')
    else:
        print('Unknown data type.')
        
    if save_fig:
        plt.savefig(out_path)


def plot_behavior_traces(
    y_obs,
    y_pred,
    behavior_type,
    data_type,
    save_fig=False,
    out_path=None
):
    locs = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    fig, axes = plt.subplots(6, 1, figsize=(12, 14))
    for i in range(6):
        axes[i].plot(y_obs[i*200:(i+1)*200], 
                     c='gray', linestyle='dashed', label='observed', linewidth=3)
        axes[i].plot(y_pred[i*200:(i+1)*200], 
                     c='blue', alpha=.6, label=f'{data_type}', linewidth=3)
        axes[i].set_xticks(locs, np.arange(i*9, (i+1)*9)*1.5)

    axes[0].set_title(f'{behavior_type}')
    axes[-1].set_xlabel('time (sec)')
    axes[0].legend(loc='upper left', ncol=2, fancybox=False, shadow=False, frameon=False)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(out_path)
        
        