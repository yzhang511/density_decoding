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
    to do: change facecolor to black and use 'cubehelix' cmap
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
    plt.scatter(spikes_display[:,1], spikes_display[:,2], c=spikes_display[:,3], s=1, alpha=0.1),
    plt.xlabel("x (um)")
    plt.ylabel("z (um)")
    plt.title(f'spikes collected over {len(trials_ids)} trials')
    plt.colorbar(label="ptp (amp)");


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

def plot_gmm_cluster_viz(sub_id, data, labels, labels_display, display_all_spikes=False, n_spikes_display=30_000, title=None, save_fig=False):
    '''
    '''
    fig, axes = plt.subplots(1, 2, figsize=(8,16)) 
    colors = [k for k,v in pltc.cnames.items()]
    random.shuffle(colors)
    for i in np.unique(labels_display):
        if i >= 148:
            c = int(i.copy()) // 4  # only 148 colors available for plotting
        else:
            c = int(i.copy())
        if len(data[labels == i, 0]) > 10:
            confidence_ellipse(data[labels == i, 0], data[labels == i, 1], 
                               axes[0], alpha=0.07, facecolor=colors[c], edgecolor=colors[c], zorder=0)
            confidence_ellipse(data[labels == i, 2], data[labels == i, 1], 
                               axes[1], alpha=0.07, facecolor=colors[c], edgecolor=colors[c], zorder=0)
            if display_all_spikes:
                axes[0].scatter(data[labels == i][:,0], data[labels == i][:,1], 
                            s=1, alpha=0.01, c=colors[c])
                axes[1].scatter(data[labels == i][:,2], data[labels == i][:,1], 
                            s=1, alpha=0.01, c=colors[c])
            else:
                axes[0].scatter(data[labels == i][:n_spikes_display,0], data[labels == i][:n_spikes_display,1], 
                                s=1, alpha=0.01, c=colors[c])
                axes[1].scatter(data[labels == i][:n_spikes_display,2], data[labels == i][:n_spikes_display,1], 
                            s=1, alpha=0.01, c=colors[c])
            axes[0].set_xlabel('x (um)')
            axes[0].set_ylabel('z (um)')
            axes[0].set_title(f'{title}')
            axes[1].set_xlabel('max ptp (amp)')
            axes[1].set_ylabel('z (um)')
            axes[1].set_title(f'n_gaussians = {len(np.unique(labels_display))}')

    for ax in ['top','bottom','left','right']:
        axes[0].spines[ax].set_linewidth(1.5)
        axes[1].spines[ax].set_linewidth(1.5)

    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'../data/{sub_id}/plots/{title}_MoG_{len(np.unique(labels_display))}.png', dpi=200)
        plt.show()
    else:
        plt.show()
        

def plot_gmm_cluster_bounds(sub_id, data, labels, labels_display, display_all_spikes=False, n_spikes_display=30_000, figure_size=(8,16), title=None, save_fig=False):
    '''
    '''
    fig, axes = plt.subplots(1, 2, figsize=figure_size) 
    colors = [k for k,v in pltc.cnames.items()]
    random.shuffle(colors)
    for i in np.unique(labels_display):
        if i >= 148:
            c = int(i.copy()) // 10  # only 148 colors available for plotting
        else:
            c = int(i.copy())
        if len(data[labels == i, 0]) > 10:
            confidence_ellipse(data[labels == i, 0], data[labels == i, 1], 
                               axes[0], alpha=1., edgecolor=colors[c], linewidth=1., zorder=0)
            confidence_ellipse(data[labels == i, 2], data[labels == i, 1], 
                               axes[1], alpha=1., edgecolor=colors[c], linewidth=1.,  zorder=0)
            if display_all_spikes:
                axes[0].scatter(data[labels == i][:,0], data[labels == i][:,1], 
                            s=.5, alpha=0.02, c=data[labels == i][:n_spikes_display,2])
                axes[1].scatter(data[labels == i][:,2], data[labels == i][:,1], 
                            s=.5, alpha=0.02, c=data[labels == i][:n_spikes_display,2])
            else:
                axes[0].scatter(data[labels == i][:n_spikes_display,0], data[labels == i][:n_spikes_display,1], 
                                s=.5, alpha=0.02, c=data[labels == i][:n_spikes_display,2])
                axes[1].scatter(data[labels == i][:n_spikes_display,2], data[labels == i][:n_spikes_display,1], 
                                s=.5, alpha=0.02, c=data[labels == i][:n_spikes_display,2])
            axes[0].set_xlim(-100, 175)
            axes[0].set_ylim(-50, 4000)
            axes[0].set_xlabel('x (um)')
            axes[0].set_ylabel('z (um)')
            axes[0].set_title(f'{title}')
            axes[1].set_xlim(0, 60)
            axes[1].set_ylim(-50, 4000)
            axes[1].set_xlabel('max ptp (amp)')
            axes[1].set_ylabel('z (um)')
            axes[1].set_title(f'n_gaussians = {len(np.unique(labels_display))}')

    for ax in ['top','bottom','left','right']:
        axes[0].spines[ax].set_linewidth(1.5)
        axes[1].spines[ax].set_linewidth(1.5)

    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'../data/{sub_id}/plots/{title}_MoG_{len(np.unique(labels_display))}.png', dpi=200)
        plt.show()
    else:
        plt.show()
        
        
def plot_gaussian_mixtures(sub_id, data, labels, labels_display, display_all_spikes=False, n_spikes_display=30_000, figure_size=(8,16), title=None, save_fig=False):
    '''
    to do: plot our MoG and kilosort MoG side-by-side.
    '''
    fig, axes = plt.subplots(1, 2, figsize=figure_size) 
    colors = [k for k,v in pltc.cnames.items()]
    random.shuffle(colors)
    for i in np.unique(labels_display):
        if i >= 148:
            c = int(i.copy()) // 10  # only 148 colors available for plotting
        else:
            c = int(i.copy())
        if len(data[labels == i, 0]) > 10:
            if display_all_spikes:
                axes[0].scatter(data[labels == i][:,0], data[labels == i][:,1], 
                            s=.5, alpha=0.05, c=colors[c])
                axes[1].scatter(data[labels == i][:,2], data[labels == i][:,1], 
                            s=.5, alpha=0.05, c=colors[c])
            else:
                axes[0].scatter(data[labels == i][:n_spikes_display,0], data[labels == i][:n_spikes_display,1], 
                                s=.5, alpha=0.05, c=colors[c])
                axes[1].scatter(data[labels == i][:n_spikes_display,2], data[labels == i][:n_spikes_display,1], 
                                s=.5, alpha=0.05, c=colors[c])
            axes[0].set_xlim(-100, 175)
            axes[0].set_ylim(-50, 4000)
            axes[0].set_xlabel('x (um)')
            axes[0].set_ylabel('z (um)')
            axes[0].set_title(f'{title}')
            axes[0].set_facecolor('black')
            axes[1].set_xlim(0, 60)
            axes[1].set_ylim(-50, 4000)
            axes[1].set_xlabel('max ptp (amp)')
            axes[1].set_ylabel('z (um)')
            axes[1].set_facecolor('black')
            axes[1].set_title(f'n_gaussians = {len(np.unique(labels_display))}')

    for ax in ['top','bottom','left','right']:
        axes[0].spines[ax].set_linewidth(1.5)
        axes[1].spines[ax].set_linewidth(1.5)

    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'../data/{sub_id}/plots/{title}_MoG_{len(np.unique(labels_display))}.png', dpi=200)
        plt.show()
    else:
        plt.show()
    

def define_box_properties(plot_name, color_code, label):
    '''
    
    '''
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    plt.plot([], c=color_code, label=label)
    plt.legend()
    
    
def plot_compare_decoder_boxplots(rootpath, sub_id, behave_type, metric_type, rois, n_folds, add_smooth=False, fig_size=(15,5), font_size=15, save_fig=False):
    '''
    to do: optimize this code and reduce code redundancy.
    '''
    if add_smooth:
        smooth_type = '_tpca'
    else:
        smooth_type = ''
        
    non_clusterless_decode_results = np.load(
        f'{rootpath}/{sub_id}/decode_results/non_clusterless{smooth_type}_decode_results.npy', 
        allow_pickle=True).item()
    
    clusterless_decode_results = np.load(
        f'{rootpath}/{sub_id}/decode_results/clusterless{smooth_type}_decode_results.npy', 
        allow_pickle=True).item()
    
    regional_decode_results = {
    rois[0]: np.load(f'{rootpath}/{sub_id}/decode_results/{rois[0]}{smooth_type}_decode_results.npy', 
                     allow_pickle=True).item(),
    rois[1]: np.load(f'{rootpath}/{sub_id}/decode_results/{rois[1]}{smooth_type}_decode_results.npy', 
                     allow_pickle=True).item(),     
    rois[2]: np.load(f'{rootpath}/{sub_id}/decode_results/{rois[2]}{smooth_type}_decode_results.npy', 
                     allow_pickle=True).item(),
    rois[3]: np.load(f'{rootpath}/{sub_id}/decode_results/{rois[3]}{smooth_type}_decode_results.npy', 
                     allow_pickle=True).item(),
    rois[4]: np.load(f'{rootpath}/{sub_id}/decode_results/{rois[4]}{smooth_type}_decode_results.npy', 
                     allow_pickle=True).item()
    }

    if metric_type == 'accuracy':
        idx = 0
    elif metric_type == 'auc':
        idx = 1
        
    data_type = 'good units'
    good_units = [non_clusterless_decode_results[behave_type][data_type][idx],
                     regional_decode_results[rois[0]][behave_type][data_type][idx],
                     regional_decode_results[rois[1]][behave_type][data_type][idx],
                     regional_decode_results[rois[2]][behave_type][data_type][idx],
                     regional_decode_results[rois[3]][behave_type][data_type][idx],
                     regional_decode_results[rois[4]][behave_type][data_type][idx]]
    data_type = 'kilosort unsorted'
    ks_unsorted = [non_clusterless_decode_results[behave_type][data_type][idx],
                     regional_decode_results[rois[0]][behave_type][data_type][idx],
                     regional_decode_results[rois[1]][behave_type][data_type][idx],
                     regional_decode_results[rois[2]][behave_type][data_type][idx],
                     regional_decode_results[rois[3]][behave_type][data_type][idx],
                     regional_decode_results[rois[4]][behave_type][data_type][idx]]
    data_type = 'sorted'
    sorted = [non_clusterless_decode_results[behave_type][data_type][idx],
                 regional_decode_results[rois[0]][behave_type][data_type][idx],
                 regional_decode_results[rois[1]][behave_type][data_type][idx],
                 regional_decode_results[rois[2]][behave_type][data_type][idx],
                 regional_decode_results[rois[3]][behave_type][data_type][idx],
                 regional_decode_results[rois[4]][behave_type][data_type][idx]]
    data_type = 'unsorted'
    unsorted = [non_clusterless_decode_results[behave_type][data_type][idx],
                     regional_decode_results[rois[0]][behave_type][data_type][idx],
                     regional_decode_results[rois[1]][behave_type][data_type][idx],
                     regional_decode_results[rois[2]][behave_type][data_type][idx],
                     regional_decode_results[rois[3]][behave_type][data_type][idx],
                     regional_decode_results[rois[4]][behave_type][data_type][idx]]
    data_type = 'clusterless'
    clusterless = [clusterless_decode_results[behave_type][data_type][idx],
                     regional_decode_results[rois[0]][behave_type][data_type][idx],
                     regional_decode_results[rois[1]][behave_type][data_type][idx],
                     regional_decode_results[rois[2]][behave_type][data_type][idx],
                     regional_decode_results[rois[3]][behave_type][data_type][idx],
                     regional_decode_results[rois[4]][behave_type][data_type][idx]]

    ticks = rois.copy(); ticks.insert(0, 'all')
    colors = ['gray', 'skyblue', 'teal', 'royalblue', 'coral']
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams.update({'font.size': font_size})
    
    good_units_plot = plt.boxplot(good_units, positions=np.array(np.arange(len(good_units)))*2.0-0.4, widths=0.15, showfliers=False)
    plt.scatter(x=np.repeat(np.array(np.arange(len(good_units))).reshape(-1,1)*2.0-0.4, n_folds, axis=1), 
                y=good_units, c=colors[0], s=10)
    
    ks_unsorted_plot = plt.boxplot(ks_unsorted, positions=np.array(np.arange(len(ks_unsorted)))*2.0-0.2, widths=0.15, showfliers=False)
    plt.scatter(x=np.repeat(np.array(np.arange(len(ks_unsorted))).reshape(-1,1)*2.0-0.2, n_folds, axis=1), 
                y=ks_unsorted, c=colors[1], s=10)

    sorted_plot = plt.boxplot(sorted, positions=np.array(np.arange(len(sorted)))*2.0, widths=0.15, showfliers=False)
    plt.scatter(x=np.repeat(np.array(np.arange(len(sorted))).reshape(-1,1)*2.0, n_folds, axis=1), 
                y=sorted, c=colors[2], s=10)
    
    unsorted_plot = plt.boxplot(unsorted, positions=np.array(np.arange(len(unsorted)))*2.0+0.2, widths=0.15, showfliers=False)
    plt.scatter(x=np.repeat(np.array(np.arange(len(unsorted))).reshape(-1,1)*2.0+0.2, n_folds, axis=1), 
                y=unsorted, c=colors[3], s=10)
    
    clusterless_plot = plt.boxplot(clusterless, positions=np.array(np.arange(len(clusterless)))*2+0.4, widths=0.15, showfliers=False)
    plt.scatter(x=np.repeat(np.array(np.arange(len(clusterless))).reshape(-1,1)*2+0.4, n_folds, axis=1), 
                y=clusterless, c=colors[4], s=10)
        
    if add_smooth:
        add_name = ' + tpca'
    else:
        add_name = ''
    define_box_properties(good_units_plot, colors[0], 'good sorted units' + add_name)
    define_box_properties(ks_unsorted_plot, colors[1], 'kilosort thresholded' + add_name)
    define_box_properties(sorted_plot, colors[2], 'all sorted units' + add_name)
    define_box_properties(unsorted_plot, colors[3], 'thresholded' + add_name)
    define_box_properties(clusterless_plot, colors[4], 'clusterless' + add_name)

    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylabel(metric_type);
    plt.title('');
    if save_fig:
        plt.savefig(f'{rootpath}/{sub_id}/plots/compare{smooth_type}_decoders_{behave_type}_{metric_type}.png', dpi=200)
        plt.show()
    else:
        plt.show()
    
    

def plot_static_behavior_traces(rootpath, sub_id, choices, stimuli, fig_size=(12,3), font_size=15, save_fig=False):
    '''
    '''
    
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams.update({'font.size': font_size})
    plt.plot(stimuli, c='gray', label='stimulus', linewidth=2)
    plt.eventplot(np.where(choices.argmax(1)), colors='red', lineoffsets=1, linewidth=1, linelengths=0.1, label='right')
    plt.eventplot(np.where(choices.argmax(1)==0), colors='green', lineoffsets=-1, linewidth=1, linelengths=0.1, label='left')
    plt.axhline(y=0, color='blue', linestyle='--')
    plt.legend(loc=0, prop={'size': 15});
    plt.ylabel('stimulus');
    plt.xlabel('trial number');  # to do: change to real time units
    plt.title("");
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{rootpath}/{sub_id}/plots/static_behavior_traces.png', dpi=200)
        plt.show()
    else:
        plt.show()
    

def plot_compare_obs_pred_static_behavior_traces(rootpath, sub_id, behave_type, data_type, trial_ids, obs_behaviors, pred_behaviors, fig_size=(12,3), font_size=15, save_fig=False):
    '''
    '''
    
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams.update({'font.size': font_size})
    plt.plot(obs_behaviors[np.argsort(trial_ids)], c='gray', linestyle='dashed', label='observed ' + behave_type)
    plt.plot(pred_behaviors[np.argsort(trial_ids)], c='blue', alpha=.6, label='predicted ' + behave_type)
    plt.legend(loc=2, prop={'size': 15});
    plt.ylabel(behave_type);
    plt.xlabel('trial number');
    plt.legend(fontsize=12);
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{rootpath}/{sub_id}/plots/{data_type}_obs_pred_{behave_type}_traces.png', dpi=200)
        plt.show()
    else:
        plt.show()

