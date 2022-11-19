import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas


def load_neural_data(
    pid, 
    trial_data_path,
    neural_data_path,
    behavior_data_path,
    keep_active_trials=True, 
    roi='all',
    kilosort=False,
    triage=False,
    good_units=False,
    thresholding=False,
):
    '''
    spike_times (seconds)
    stimulus_onset_times (seconds)
    '''
    
    pw = 'international'
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', password=pw, silent=True)
    ba = AllenAtlas()
    eid, probe = one.pid2eid(pid)
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    print(f'pid: {pid}')
    print(f'eid: {eid}')
    
    trials = one.load_object(eid, 'trials', collection='alf')
    stimulus_onset_times = trials['stimOn_times']
    np1_channel_map = np.load(f'{neural_data_path}/np1_channel_map.npy')
    
    if kilosort:
        data_path = neural_data_path + '/kilosort_localizations' 
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
        if good_units:
            good_clusters = clusters['cluster_id'][clusters['label'] == 1]
            print(f'found {len(good_clusters)} good ibl units ..')
        if os.path.exists(data_path):
            spike_train = \
                np.load(f'{data_path}/aligned_spike_train.npy')
            spike_index = \
                np.load(f'{data_path}/aligned_spike_index.npy') 
            localization_results = \
                np.load(f'{data_path}/aligned_localizations.npy')
            maxptp = \
                np.load(f'{data_path}/aligned_maxptp.npy')             
            x, _, _, z, _ = localization_results.T
            localization = True
        else:
            spike_times = sl.samples2times(spikes.times, direction='reverse')
            spike_channels = [clusters.channels[i] for i in spikes.clusters]
            spike_index = np.c_[spike_times, spike_channels]
            spike_train = np.c_[spike_times, spikes.clusters]   
            localization = False
    else:
        data_path = neural_data_path + '/subtraction_results_threshold_5'
        spike_index = \
            np.load(f'{data_path}/spike_index.npy') 
        localization_results = \
            np.load(f'{data_path}/localization_results.npy')
        x, z, maxptp = localization_results.T
        localization = True
        
    if triage:
        triage_results = \
            np.load(f'{data_path}/triage_results/triage_results.npy')
        triage_idx_keep = \
            np.load(f'{data_path}/triage_results/triage_idx_keep.npy')
        triage_low_ptp_filter = \
            np.load(f'{data_path}/triage_results/triage_low_ptp_filter.npy')
        if kilosort:
            spike_train = spike_train[triage_low_ptp_filter]
            spike_train = spike_train[triage_idx_keep]
        spike_index = spike_index[triage_low_ptp_filter]
        spike_index = spike_index[triage_idx_keep]
        x, z, maxptp = triage_results.T
        localization = True
        
    if keep_active_trials:
        active_trials_ids = np.load(f'{behavior_data_path}/{eid}_trials.npy')
        stimulus_onset_times = stimulus_onset_times[active_trials_ids]
        
    n_trials = stimulus_onset_times.shape[0]
    print(f'1st trial stim on time: {stimulus_onset_times[0]:.2f}, last trial stim on time {stimulus_onset_times[-1]:.2f}') 
    
    spike_times, spike_channels = spike_index.T
    spike_times = sl.samples2times(spike_times)
    
    if kilosort:
        _, spike_clusters = spike_train.T
        sorted = np.c_[spike_times, spike_clusters]
        
    if localization:        
        unsorted = np.c_[spike_times, spike_channels, x, z, maxptp]  
    elif thresholding:
        unsorted = np.c_[spike_times, spike_channels]

    if roi != 'all':
        spikes, clusters, channels = sl.load_spike_sorting()
        clusters = sl.merge_clusters(spikes, clusters, channels)
        clusters_channels = clusters['channels']
        clusters_rois = clusters['acronym']
        channels_rois = channels['acronym']
        clusters_rois = np.c_[np.arange(clusters_rois.shape[0]), clusters_rois]
        channels_rois = np.c_[np.arange(channels_rois.shape[0]), channels_rois]
        valid_clusters = clusters_rois[[roi in x.lower() for x in clusters_rois[:,-1]], 0]
        valid_clusters = np.unique(valid_clusters).astype(int)
        valid_channels = channels_rois[[roi in x.lower() for x in channels_rois[:,-1]], 0]
        valid_channels = np.unique(valid_channels).astype(int)
        print(f'found {len(valid_clusters)} neurons in region {roi} ...')
        print(f'found {len(valid_channels)} channels in region {roi} ...')
        
        if kilosort:
            sorted_regional = []
            for cluster in valid_clusters:
                sorted_regional.append(sorted[sorted[:,1] == cluster])
            sorted = np.vstack(sorted_regional)
            if good_units:
                good_sorted_regional = []
                for cluster in np.intersect1d(valid_clusters, good_clusters):
                    good_sorted_regional.append(sorted[sorted[:,1] == cluster])
                    good_sorted = np.vstack(good_sorted_regional)
        
        if np.logical_or(localization, thresholding):
            unsorted_regional = []
            for i in valid_channels:
                unsorted_regional.append(unsorted[unsorted[:,1] == i])
            unsorted = np.vstack(unsorted_regional)
            
    unsorted_trials = []
    if np.logical_or(localization, thresholding):
        for i in range(n_trials):
            mask = np.logical_and( unsorted[:,0] >= stimulus_onset_times[i]-0.5,   
                                 unsorted[:,0] <= stimulus_onset_times[i]+1 )  
            trial = unsorted[mask,:]
            unsorted_trials.append(trial)
            
    if kilosort:
        sorted_trials = []
        for i in range(n_trials):
            mask = np.logical_and( sorted[:,0] >= stimulus_onset_times[i]-0.5,   
                                 sorted[:,0] <= stimulus_onset_times[i]+1 )  
            trial = sorted[mask,:]
            sorted_trials.append(trial)
        if good_units:
            good_sorted = np.vstack([sorted[sorted[:,1] == cluster] for cluster in good_clusters])
            tmp = pd.DataFrame({'time': good_sorted[:,0], 'old_unit': good_sorted[:,1].astype(int)})
            tmp["old_unit"] = tmp["old_unit"].astype("category")
            tmp["new_unit"] = pd.factorize(tmp.old_unit)[0]
            good_sorted = np.array(tmp[['time','new_unit']])
            good_sorted_trials = []
            for i in range(n_trials):
                mask = np.logical_and( good_sorted[:,0] >= stimulus_onset_times[i]-0.5,   
                                     good_sorted[:,0] <= stimulus_onset_times[i]+1 )  
                trial = good_sorted[mask,:]
                good_sorted_trials.append(trial)
            return sorted_trials, good_sorted_trials, unsorted_trials, stimulus_onset_times, np1_channel_map
        else:
            return sorted_trials, unsorted_trials, stimulus_onset_times, np1_channel_map
    else:
        return unsorted_trials, stimulus_onset_times, np1_channel_map
        
    

def compute_neural_activity(
    data, 
    stimulus_onset_times,
    data_type, 
    n_time_bins=30, 
    regional=False  
):
    '''
    
    '''
    binning = np.arange(0, 1.5, step=(1.5 - 0)/n_time_bins)
    n_trials = stimulus_onset_times.shape[0]
    neural_data = []
    
    if data_type=='clusterless':
        spike_times, spike_labels, spike_probs = data
        n_gaussians = len(np.unique(spike_labels))
        spike_probs = spike_probs[:, np.unique(spike_labels)]
        spike_train = np.c_[spike_times, spike_labels, spike_probs]

        for i in range(n_trials):
            mask = np.logical_and(spike_train[:,0] >= stimulus_onset_times[i]-0.5,
                                  spike_train[:,0] <= stimulus_onset_times[i]+1
                                 )
            trial = spike_train[mask,:]
            trial[:,0] = trial[:,0] - trial[:,0].min()
            time_bins = np.digitize(trial[:,0], binning, right=False)-1
            time_bins_lst = []
            for t in range(n_time_bins):
                time_bin = trial[time_bins == t, 2:]
                gmm_weights_lst = np.zeros(n_gaussians)
                for k in range(n_gaussians):
                    gmm_weights_lst[k] = np.sum(time_bin[:,k])
                time_bins_lst.append(gmm_weights_lst)
            neural_data.append(np.array(time_bins_lst))
        neural_data = np.array(neural_data).transpose(0,2,1)
    
    else:
        spike_times, spike_units = data
        spike_train = np.c_[spike_times, spike_units]
            
        if regional:
            n_units = len(np.unique(spike_units))
            tmp = pd.DataFrame({'time': spike_times, 'old_unit': spike_units.astype(int)})
            tmp['old_unit'] = tmp['old_unit'].astype("category")
            tmp['new_unit'] = pd.factorize(tmp.old_unit)[0]
            spike_train = np.array(tmp[['time','new_unit']])
        else:
            n_units = spike_units.max().astype(int)+1

        for i in range(n_trials):
            mask = np.logical_and(spike_train[:,0] >= stimulus_onset_times[i]-0.5,
                                  spike_train[:,0] <= stimulus_onset_times[i]+1 )
            trial = spike_train[mask,:]
            trial[:,0] = trial[:,0] - trial[:,0].min()
            units = trial[:,1].astype(int)
            time_bins = np.digitize(trial[:,0], binning, right=False)-1
            spike_count = np.zeros([n_units, n_time_bins])
            np.add.at(spike_count, (units, time_bins), 1) 
            neural_data.append(spike_count)
        neural_data = np.array(neural_data)
    
    return neural_data


def load_behaviors_data(path, pid):
    '''
    
    '''
    pw = 'international'
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', password=pw, silent=True)
    eid, probe = one.pid2eid(pid)
    behave_dict = np.load(f'{path}/{eid}_feature.npy')
    return behave_dict

    
def preprocess_static_behaviors(behave_dict):
    '''
    extract choices, stimuli, rewards and priors.
    to do: use 'behave_idx_dict' to select behaviors instead of hard-coding.
    '''
    
    choices = behave_dict[:,:,:,23:25].sum(2)[0,:,:]
    stimuli = behave_dict[:,:,:,19:21].sum(2)[0,:,:]
    rewards = behave_dict[:,:,:,25:27].sum(2)[0,:,:]
    priors = behave_dict[0,:,0,28:29]
        
    print('choices left: %.3f, right: %.3f'%((choices.sum(0)[0]/choices.shape[0]), 
                                             (choices.sum(0)[1]/choices.shape[0])))
    print('stimuli left: %.3f, right: %.3f'%((np.sum(stimuli.argmax(1)==1)/stimuli.shape[0]), 
                                   (np.sum(stimuli.argmax(1)==0)/stimuli.shape[0])))
    print('reward wrong: %.3f, correct: %.3f'%((rewards.sum(0)[0]/rewards.shape[0]), 
                                               (rewards.sum(0)[1]/rewards.shape[0])))
    
    # transform stimulus for plotting
    transformed_stimuli = []
    for s in stimuli:
        if s.argmax()==1:
            transformed_stimuli.append(-1*s.sum())
        else:
            transformed_stimuli.append(s.sum())
    transformed_stimuli = np.array(transformed_stimuli)
    
    # convert stimulus to a categeorical variable for decoding
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(transformed_stimuli.reshape(-1,1))
    one_hot_stimuli = enc.transform(transformed_stimuli.reshape(-1,1)).toarray()

    return choices, stimuli, transformed_stimuli, one_hot_stimuli, enc.categories_, rewards, priors
 
    
def inverse_transform_stimulus(transformed_stimuli, enc_categories):
    '''
    '''
    
    enc_dict = {}
    for i in np.arange(0, len(enc_categories[0])):
        enc_dict.update({i: enc_categories[0][i]})
    print(enc_dict)
    
    original_stimuli = np.zeros(len(transformed_stimuli))
    for i, s in enumerate(transformed_stimuli):
        original_stimuli[i] = enc_dict[s]
    
    return original_stimuli
    
    
def preprocess_dynamic_behaviors(behave_dict):
    '''
    to do: use 'behave_idx_dict' to select behaviors instead of hard-coding.
    '''
    
    motion_energy = behave_dict[0,:,:,18]
    wheel_velocity = behave_dict[0,:,:,27]
    wheel_speed = np.abs(wheel_velocity)
    paw_speed = behave_dict[0,:,:,15]
    nose_speed = behave_dict[0,:,:,16]
    pupil_diameter = behave_dict[0,:,:,17]
    
    return motion_energy, wheel_velocity, wheel_speed, paw_speed, nose_speed, pupil_diameter