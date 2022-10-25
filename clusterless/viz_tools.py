import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def plot_drift_map(spikes_indices, spikes_features, n_spikes_display=1000_000, samp_freq=30_000):
    '''
    to do: add kilosort drift map
    to do: pass aesthetic config
    '''
    offset = spikes_indices[:,0] / samp_freq
    spikes_times = (spikes_indices[:,0] - offset) / samp_freq

    log_ptp = spikes_features[:,4].copy()
    log_ptp = np.log(log_ptp+1)
    ptp_rescaled = (log_ptp - log_ptp.min())/(log_ptp.max() - log_ptp.min())
    ptp_rescaled[ptp_rescaled<0.2] = 0.2    #change bounds so that it looks more flat
    ptp_rescaled[ptp_rescaled>0.6] = 0.6
    ptp_rescaled = (ptp_rescaled - ptp_rescaled.min())/(ptp_rescaled.max() - ptp_rescaled.min())

    vir = plt.cm.get_cmap('viridis')
    color_arr = vir(ptp_rescaled)
    color_arr[:, 3] = ptp_rescaled

    plt.figure(figsize = (20, 10))
    plt.scatter(spikes_times[:n_spikes_display], spikes_features[:,2][:n_spikes_display], s=5, c=color_arr[:n_spikes_display])
    plt.xlabel('time (s)', fontsize = 20)
    plt.ylabel('depth (um)', fontsize = 20)
    plt.margins(x=0)
    plt.margins(y=0)
    plt.title('unsorted drift map', fontsize = 20)
    plt.show()
    
    
def plot_spikes_features(trials, trials_ids):
    '''
    to do: label how many minutes are used to collect the spikes
    to do: pass aesthetic config
    '''
    spikes_display = np.vstack([trials[i] for i in trials_ids])
    plt.figure(figsize = (4, 8))
    plt.scatter(spikes_display[:,1], spikes_display[:,2], c=spikes_display[:,3], s=10, alpha=0.5),
    plt.xlabel("x (um)")
    plt.ylabel("z (um)")
    plt.title(f'spikes collected over {len(trials_ids)} trials')
    plt.colorbar(label="ptp (amp)");


def plot_static_behavior_traces():
    return


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    '''
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    '''
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_gmm_cluster_viz(data, labels, labels_display, n_spikes_display=30_000):
    '''
    to do: fix bug in num of gaussians to display (repeated labels after split)
    '''
    fig, axes = plt.subplots(1, 2, figsize=(8,16)) 
    colors = [k for k,v in pltc.cnames.items()]
    random.shuffle(colors)
    for i in np.unique(labels_display):
        if i >= 148:
            c = i // 4  # only 148 colors available for plotting
        else:
            c = i.copy()
        if len(data[labels == i, 0]) == 0:
            continue
        confidence_ellipse(data[labels == i, 0], data[labels == i, 1], 
                           axes[0], alpha=0.07, facecolor=colors[c], edgecolor=colors[c], zorder=0)
        axes[0].scatter(data[labels == i][:n_spikes_display,0], data[labels == i][:n_spikes_display,1], 
                        s=1, alpha=0.01, c=colors[c])
        axes[0].set_xlabel('x (um)')
        axes[0].set_ylabel('z (um)')
        axes[0].set_title('MoG')

        confidence_ellipse(data[labels == i, 2], data[labels == i, 1], 
                           axes[1], alpha=0.07, facecolor=colors[c], edgecolor=colors[c], zorder=0)
        axes[1].scatter(data[labels == i][:n_spikes_display,2], data[labels == i][:n_spikes_display,1], 
                        s=1, alpha=0.01, c=colors[c])
        axes[1].set_xlabel('max ptp (amp)')
        axes[1].set_ylabel('z (um)')
        axes[1].set_title(f'n_gaussians = {len(np.unique(labels_display))}')

    for ax in ['top','bottom','left','right']:
        axes[0].spines[ax].set_linewidth(1.5)
        axes[1].spines[ax].set_linewidth(1.5)

    plt.tight_layout()
    plt.show() 