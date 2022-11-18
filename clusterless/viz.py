import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def plot_drift_map(
    spike_index, 
    spike_features, 
    n_spikes=1000_000, 
    samp_freq=30_000
):
    '''
    to do: add kilosort drift map
    to do: pass aesthetic config
    to do: change facecolor to black and use 'cubehelix' cmap
    '''
    offset = spike_index[:,0] / samp_freq
    spike_times = (spike_index[:,0] - offset) / samp_freq

    _, z, maxptp = spike_features[:,2:].T
    log_ptp = maxptp.copy()
    log_ptp = np.log(log_ptp+1)
    ptp_rescaled = (log_ptp - log_ptp.min())/(log_ptp.max() - log_ptp.min())
    #change bounds so that it looks more flat
    ptp_rescaled[ptp_rescaled<0.2] = 0.2   
    ptp_rescaled[ptp_rescaled>0.6] = 0.6
    ptp_rescaled = (ptp_rescaled - ptp_rescaled.min())/(ptp_rescaled.max() - ptp_rescaled.min())

    vir = plt.cm.get_cmap('viridis') 
    color_arr = vir(ptp_rescaled)
    color_arr[:, 3] = ptp_rescaled

    plt.figure(figsize = (20, 10))
    plt.scatter(spike_times[:n_spikes], z[:n_spikes], 
                s=5, c=maxptp[:n_spikes])
    plt.xlabel('time (s)', fontsize = 20)
    plt.ylabel('depth (um)', fontsize = 20)
    plt.margins(x=0)
    plt.margins(y=0)
    plt.title('unsorted drift map', fontsize = 20)
    plt.show()
    
    
def plot_spike_features(trials, trials_ids):
    '''
    to do: label how many minutes are used to collect the spikes
    to do: pass aesthetic config
    '''
    x, z, maxptp = np.vstack([trials[i] for i in trials_ids])[:,2:].T
    plt.figure(figsize = (4, 8))
    plt.scatter(x, z, c=maxptp, s=1, alpha=0.1),
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


def plot_MoG_bounds(
    data, 
    labels, 
    labels_display, 
    np1_channel_map,
    channels=[], 
    local=False,
    plot_MoG=False,
    n_spikes=5_000, 
    figure_size=(8,16), 
    title='', 
):
    '''
    '''
    fig, axes = plt.subplots(1, 2, figsize=figure_size) 
    colors = [k for k,v in pltc.cnames.items()]
    random.shuffle(colors)
    for idx in labels_display:
        c = int(idx)
        if idx >= len(colors):
            c = c // 6  # only 148 colors available for plotting
            
        sub_data = data[labels == idx]
        x, z, maxptp = sub_data.T
        
        if len(sub_data) <= 2:
            continue
            
        if plot_MoG:
            confidence_ellipse(x, z, axes[0], alpha=1., edgecolor=colors[c], linewidth=1., zorder=0)
            confidence_ellipse(maxptp, z, axes[1], alpha=1., edgecolor=colors[c], linewidth=1., zorder=0)
    
        if local:
            axes[0].scatter(x, z, s=.5, alpha=0.2, c=maxptp)
            axes[1].scatter(maxptp, z, s=.5, alpha=0.2, c=maxptp)
        else:
            axes[0].scatter(x[:n_spikes], z[:n_spikes], 
                            s=.5, alpha=0.05, c=maxptp[:n_spikes])
            axes[1].scatter(maxptp[:n_spikes], z[:n_spikes], 
                            s=.5, alpha=0.05, c=maxptp[:n_spikes])
            
        if local:
            axes[0].set_xlim(x.min()-25, x.max()+25)
            axes[0].set_ylim(z.min()-25, z.max()+25)
            axes[1].set_xlim(0, maxptp.max()+25)
            axes[1].set_ylim(z.min()-25, z.max()+25)
        else:    
            axes[0].set_xlim(-100, 175)
            axes[0].set_ylim(-80, 4000)
            axes[1].set_xlim(0, 60)
            axes[1].set_ylim(-80, 4000)
        
    if len(channels) > 0: 
        for channel in channels:
            axes[0].plot(np1_channel_map[int(channel), 0], 
                         np1_channel_map[int(channel), 1], 
                         markersize=3, c='orange', marker="s")
                
    axes[0].set_xlabel('x (um)')
    axes[0].set_ylabel('z (um)')
    axes[0].set_title(f'{title}')
    axes[1].set_xlabel('max ptp (amp)')
    axes[1].set_ylabel('z (um)')
    if plot_MoG:
        axes[1].set_title(f'n_units = {len(np.unique(labels_display))}')

    for ax in ['top','bottom','left','right']:
        axes[0].spines[ax].set_linewidth(1)
        axes[1].spines[ax].set_linewidth(1)

    plt.tight_layout()
    plt.show()
        
        
def plot_gaussian_mixtures(
    data, 
    labels, 
    labels_display, 
    n_spikes=5_000, 
    figure_size=(8,16), 
    title=None, 
    save_fig=False, 
    fig_path=None
):
    '''
    
    '''
    fig, axes = plt.subplots(1, 2, figsize=figure_size) 
    colors = [k for k,v in pltc.cnames.items()]
    random.shuffle(colors)
    
    for idx in labels_display:
        c = int(idx)
        if idx >= len(colors):
            c = c // 6  # only 148 colors available for plotting
            
        sub_data = data[labels == idx]
        x, z, maxptp = sub_data.T
        
        if len(sub_data) <= 2:
            continue
            
        axes[0].scatter(x[:n_spikes], z[:n_spikes], s=.5, alpha=0.05, c=colors[c])
        axes[1].scatter(maxptp[:n_spikes], z[:n_spikes], s=.5, alpha=0.05, c=colors[c])
        
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
        plt.savefig(f'{fig_path}/{title}.png', dpi=200)
    else:
        plt.show()
        
        
def plot_compare_decoder_barplots(
    data_path,
    behave_type, 
    metric_type, 
    rois, 
    n_folds, 
    add_smooth=False, 
    figure_size=(15,5), 
    font_size=15, 
    title='',
    save_fig=False,
    fig_path=None,
):
    '''
    to do: optimize this code and reduce code redundancy.
    '''
    if add_smooth:
        smooth_type = '_tpca'
    else:
        smooth_type = ''
        
    all_decode_results = np.load(
        f'{data_path}/all{smooth_type}_decode_results.npy', allow_pickle=True).item()
    
    regional_decode_results = {
        rois[0]: np.load(
            f'{data_path}/{rois[0]}{smooth_type}_decode_results.npy', allow_pickle=True).item(),
        rois[1]: np.load(
            f'{data_path}/{rois[1]}{smooth_type}_decode_results.npy', allow_pickle=True).item(),     
        rois[2]: np.load(
            f'{data_path}/{rois[2]}{smooth_type}_decode_results.npy', allow_pickle=True).item(),
        rois[3]: np.load(
            f'{data_path}/{rois[3]}{smooth_type}_decode_results.npy', allow_pickle=True).item(),
        rois[4]: np.load(
            f'{data_path}/{rois[4]}{smooth_type}_decode_results.npy', allow_pickle=True).item()
    }

    if np.logical_or(metric_type == 'accuracy', metric_type == 'r2'):
        idx = 0
    elif np.logical_or(metric_type == 'auc', metric_type == 'rmse'):
        idx = 1
        
    data_type = 'sorted'
    sorted = np.array([
        all_decode_results[behave_type][data_type][idx],
        regional_decode_results[rois[0]][behave_type][data_type][idx],
        regional_decode_results[rois[1]][behave_type][data_type][idx],
        regional_decode_results[rois[2]][behave_type][data_type][idx],
        regional_decode_results[rois[3]][behave_type][data_type][idx],
        regional_decode_results[rois[4]][behave_type][data_type][idx]
    ])
    
    data_type = 'thresholded'
    thresholded = np.array([
        all_decode_results[behave_type][data_type][idx],
        regional_decode_results[rois[0]][behave_type][data_type][idx],
        regional_decode_results[rois[1]][behave_type][data_type][idx],
        regional_decode_results[rois[2]][behave_type][data_type][idx],
        regional_decode_results[rois[3]][behave_type][data_type][idx],
        regional_decode_results[rois[4]][behave_type][data_type][idx]
    ])
    
    data_type = 'clusterless'
    clusterless = np.array([
        all_decode_results[behave_type][data_type][idx],
        regional_decode_results[rois[0]][behave_type][data_type][idx],
        regional_decode_results[rois[1]][behave_type][data_type][idx],
        regional_decode_results[rois[2]][behave_type][data_type][idx],
        regional_decode_results[rois[3]][behave_type][data_type][idx],
        regional_decode_results[rois[4]][behave_type][data_type][idx]
    ])

    ticks = rois.copy(); ticks.insert(0, 'all')
    # plt.rcParams["figure.figsize"] = figure_size
    plt.rcParams.update({'font.size': font_size})

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    mins, maxs, means, stds = sorted.min(1), sorted.max(1), sorted.mean(1), sorted.std(1)
    plt.errorbar(np.arange(len(ticks))*2-.2, means, stds, 
                     fmt='.k', ecolor='teal', lw=3, label='sorted')
    plt.errorbar(np.arange(len(ticks))*2-.2, means, [means - mins, maxs - means],
                     fmt='.k', ecolor='gray', lw=1.5)
    
    mins, maxs, means, stds = thresholded.min(1), thresholded.max(1), thresholded.mean(1), thresholded.std(1)
    plt.errorbar(np.arange(len(ticks))*2, means, stds, 
                 fmt='.k', ecolor='royalblue', lw=3, label='thresholded')
    plt.errorbar(np.arange(len(ticks))*2, means, [means - mins, maxs - means],
                     fmt='.k', ecolor='gray', lw=1.5)
    
    mins, maxs, means, stds = clusterless.min(1), clusterless.max(1), clusterless.mean(1), clusterless.std(1)
    plt.errorbar(np.arange(len(ticks))*2+.2, means, stds, 
                 fmt='.k', ecolor='coral', lw=3, label='clusterless')
    plt.errorbar(np.arange(len(ticks))*2+.2, means, [means - mins, maxs - means],
                     fmt='.k', ecolor='gray', lw=1.5)
        
    if add_smooth:
        add_name = ' + tpca'
    else:
        add_name = ''
        
    plt.legend(loc='lower left')
    ax.set_xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    ax.set_xlim(-2, len(ticks)*2)
    ax.set_ylabel(metric_type);
    ax.set_title(title);
    
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    
    if save_fig:
        fig.savefig(f'{fig_path}/compare{smooth_type}_decoders_{behave_type}_{metric_type}.png', dpi=200)
    else:
        plt.show()
    

def define_box_properties(plot_name, color_code, label):
    '''
    
    '''
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    plt.plot([], c=color_code, label=label)
    plt.legend()
    
    
def plot_compare_decoder_boxplots(
    data_path,
    behave_type, 
    metric_type, 
    rois, 
    n_folds, 
    add_smooth=False, 
    figure_size=(15,5), 
    font_size=15, 
    save_fig=False,
    fig_path=None,
):
    '''
    to do: optimize this code and reduce code redundancy.
    '''
    if add_smooth:
        smooth_type = '_tpca'
    else:
        smooth_type = ''
        
    all_decode_results = np.load(
        f'{data_path}/all{smooth_type}_decode_results.npy', allow_pickle=True).item()
    
    regional_decode_results = {
        rois[0]: np.load(
            f'{data_path}/{rois[0]}{smooth_type}_decode_results.npy', allow_pickle=True).item(),
        rois[1]: np.load(
            f'{data_path}/{rois[1]}{smooth_type}_decode_results.npy', allow_pickle=True).item(),     
        rois[2]: np.load(
            f'{data_path}/{rois[2]}{smooth_type}_decode_results.npy', allow_pickle=True).item(),
        rois[3]: np.load(
            f'{data_path}/{rois[3]}{smooth_type}_decode_results.npy', allow_pickle=True).item(),
        rois[4]: np.load(
            f'{data_path}/{rois[4]}{smooth_type}_decode_results.npy', allow_pickle=True).item()
    }

    if metric_type == 'accuracy':
        idx = 0
    elif metric_type == 'auc':
        idx = 1
        
    data_type = 'sorted'
    sorted = [
        all_decode_results[behave_type][data_type][idx],
        regional_decode_results[rois[0]][behave_type][data_type][idx],
        regional_decode_results[rois[1]][behave_type][data_type][idx],
        regional_decode_results[rois[2]][behave_type][data_type][idx],
        regional_decode_results[rois[3]][behave_type][data_type][idx],
        regional_decode_results[rois[4]][behave_type][data_type][idx]
    ]
    
    data_type = 'thresholded'
    thresholded = [
        all_decode_results[behave_type][data_type][idx],
        regional_decode_results[rois[0]][behave_type][data_type][idx],
        regional_decode_results[rois[1]][behave_type][data_type][idx],
        regional_decode_results[rois[2]][behave_type][data_type][idx],
        regional_decode_results[rois[3]][behave_type][data_type][idx],
        regional_decode_results[rois[4]][behave_type][data_type][idx]
    ]
    
    data_type = 'clusterless'
    clusterless = [
        all_decode_results[behave_type][data_type][idx],
        regional_decode_results[rois[0]][behave_type][data_type][idx],
        regional_decode_results[rois[1]][behave_type][data_type][idx],
        regional_decode_results[rois[2]][behave_type][data_type][idx],
        regional_decode_results[rois[3]][behave_type][data_type][idx],
        regional_decode_results[rois[4]][behave_type][data_type][idx]
    ]

    ticks = rois.copy(); ticks.insert(0, 'all')
    colors = ['teal', 'royalblue', 'coral']
    plt.rcParams["figure.figsize"] = figure_size
    plt.rcParams.update({'font.size': font_size})

    sorted_plot = plt.boxplot(
        sorted, 
        positions=np.array(np.arange(len(sorted)))*2.0-0.2, 
        widths=0.15, 
        showfliers=False)
    
    plt.scatter(x=np.repeat(np.array(np.arange(len(sorted))).reshape(-1,1)*2.0-0.2, n_folds, axis=1), 
                y=sorted, c=colors[0], s=10)
    
    thresholded_plot = plt.boxplot(
        thresholded, 
        positions=np.array(np.arange(len(thresholded)))*2.0, 
        widths=0.15, 
        showfliers=False)
    
    plt.scatter(x=np.repeat(np.array(np.arange(len(thresholded))).reshape(-1,1)*2.0, n_folds, axis=1), 
                y=thresholded, c=colors[1], s=10)
    
    clusterless_plot = plt.boxplot(
        clusterless, 
        positions=np.array(np.arange(len(clusterless)))*2+0.2, 
        widths=0.15, 
        showfliers=False)
    
    plt.scatter(x=np.repeat(np.array(np.arange(len(clusterless))).reshape(-1,1)*2+0.2, n_folds, axis=1), 
                y=clusterless, c=colors[2], s=10)
        
    if add_smooth:
        add_name = ' + tpca'
    else:
        add_name = ''
        
    define_box_properties(sorted_plot, colors[0], 'sorted' + add_name)
    define_box_properties(thresholded_plot, colors[1], 'thresholded' + add_name)
    define_box_properties(clusterless_plot, colors[2], 'clusterless' + add_name)

    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylabel(metric_type);
    plt.title(behave_type);
    
    if save_fig:
        plt.savefig(f'{fig_path}/compare{smooth_type}_decoders_{behave_type}_{metric_type}.png', dpi=200)
    else:
        plt.show()
    
    

def plot_static_behavior_traces(
    choices, 
    stimuli, 
    figure_size=(12,3), 
    font_size=15, 
    save_fig=False,
    fig_path=None
):
    '''
    '''
    plt.rcParams["figure.figsize"] = figure_size
    plt.rcParams.update({'font.size': font_size})
    plt.plot(stimuli, c='gray', label='stimulus', linewidth=2)
    plt.eventplot(np.where(choices.argmax(1)), colors='red', 
                  lineoffsets=1, linewidth=1, linelengths=0.1, label='right')
    plt.eventplot(np.where(choices.argmax(1)==0), colors='green', 
                  lineoffsets=-1, linewidth=1, linelengths=0.1, label='left')
    plt.axhline(y=0, color='blue', linestyle='--')
    plt.legend(loc=0, prop={'size': 15});
    plt.ylabel('stimulus');
    plt.xlabel('trial number');  # to do: change to real time units
    plt.title("");
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{fig_path}/static_behavior_traces.png', dpi=200)
    else:
        plt.show()
    

def plot_compare_obs_pred_static_behavior_traces(
    behave_type, 
    data_type, 
    trial_ids, 
    obs_behaviors, 
    pred_behaviors, 
    figure_size=(12,3), 
    font_size=15, 
    save_fig=False,
    fig_path=None
):
    '''
    '''
    plt.rcParams["figure.figsize"] = figure_size
    plt.rcParams.update({'font.size': font_size})
    plt.plot(obs_behaviors[np.argsort(trial_ids)], c='gray', 
             linestyle='dashed', label='observed ' + behave_type)
    plt.plot(pred_behaviors[np.argsort(trial_ids)], c='blue', 
             alpha=.6, label='predicted ' + behave_type)
    plt.legend(loc=2, prop={'size': 15});
    plt.ylabel(behave_type);
    plt.xlabel('trial number');
    plt.legend(fontsize=12);
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{fig_path}/{data_type}_obs_pred_{behave_type}_traces.png', dpi=200)
    else:
        plt.show()

