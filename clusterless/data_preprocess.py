import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_unsorted_data(rootpath, sub_id, roi='all', keep_active_trials=True, samp_freq=30_000):
    '''
    
    '''
    
    spikes_indices = np.load(f'{rootpath}/{sub_id}/unsorted/spikes_indices.npy')        # unit: time samples
    spikes_features = np.load(f'{rootpath}/{sub_id}/unsorted/spikes_features.npy')
    np1_channel_map = np.load(f'{rootpath}/{sub_id}/misc/np1_channel_map.npy')
    stimulus_onset_times = np.load(f'{rootpath}/{sub_id}/misc/stimulus_onset_times.npy') # unit: seconds
    
    if keep_active_trials:
        active_trials_ids = np.load(f'{rootpath}/{sub_id}/behaviors/active_trials_ids.npy')
        stimulus_onset_times = stimulus_onset_times[active_trials_ids]
        
    if roi != 'all':
        clusters_channels = np.load(f'{rootpath}/{sub_id}/sorted/clusters_channels.npy', allow_pickle=True)
        channels_rois = np.load(f'{rootpath}/{sub_id}/sorted/channels_rois.npy', allow_pickle=True)
        channels_rois = np.vstack([np.arange(384), channels_rois]).transpose()
        valid_channels = channels_rois[[roi in x for x in channels_rois[:,-1]], 0]
        valid_channels = np.unique(valid_channels).astype(int)
        print(f'found {len(valid_channels)} channels in roi {roi}')
        
        # add z_reg later 
        unsorted = np.concatenate([spikes_indices, spikes_features[:,[0,2,4]]], axis=1)
        regional = []
        for i in valid_channels:
            regional.append(unsorted[unsorted[:,1] == i])
        regional = np.vstack(regional)
        trials = []
        for i in range(stimulus_onset_times.shape[0]):
            mask = np.logical_and(regional[:,0] >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,   
                             regional[:,0] <= stimulus_onset_times[i]*samp_freq+samp_freq ) 
            trial = regional[mask,:]
            trials.append(trial[:,[0,2,3,4]]) 
        return trials
    
    else:
        # add z_reg later 
        unsorted = np.concatenate([spikes_indices[:,0].reshape(-1,1), spikes_features[:,[0,2,4]]], axis=1)
        trials = []
        for i in range(stimulus_onset_times.shape[0]):
            mask = np.logical_and(unsorted[:,0] >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,   
                                 unsorted[:,0] <= stimulus_onset_times[i]*samp_freq+samp_freq )        # 1.5 secs / trial
            trial = unsorted[mask,:]
            # trial[:,0] = (trial[:,0] - trial[:,0].min()) / samp_freq
            trials.append(trial)
        return spikes_indices, spikes_features, np1_channel_map, stimulus_onset_times, unsorted, trials
    
    
    
def load_kilosort_sorted_data(rootpath, sub_id, roi='all', keep_active_trials = True, samp_freq=30_000):
    '''
    
    '''    
    spikes_times = np.load(f'{rootpath}/{sub_id}/sorted/spikes_times.npy')
    spikes_clusters = np.load(f'{rootpath}/{sub_id}/sorted/spikes_clusters.npy')
    spikes_amps = np.load(f'{rootpath}/{sub_id}/sorted/spikes_amps.npy')
    spikes_depths = np.load(f'{rootpath}/{sub_id}/sorted/spikes_depths.npy')
    stimulus_onset_times = np.load(f'{rootpath}/{sub_id}/misc/stimulus_onset_times.npy') # unit: seconds
    
    if keep_active_trials:
        active_trials_ids = np.load(f'{rootpath}/{sub_id}/behaviors/active_trials_ids.npy')
        stimulus_onset_times = stimulus_onset_times[active_trials_ids]
    
    sorted = np.concatenate([spikes_times.reshape(-1,1), spikes_clusters.reshape(-1,1)], axis=1)
    
    if roi != 'all':
        clusters_channels = np.load(f'{rootpath}/{sub_id}/sorted/clusters_channels.npy', allow_pickle=True)
        channels_rois = np.load(f'{rootpath}/{sub_id}/sorted/channels_rois.npy', allow_pickle=True)
        channels_rois = np.vstack([np.arange(384), channels_rois]).transpose()
        valid_channels = channels_rois[[roi in x for x in channels_rois[:,-1]], 0]
        valid_channels = np.unique(valid_channels).astype(int)
        valid_units = []
        for c in valid_channels:
            units_per_channel = list(np.where(clusters_channels == c)[0])
            if len(units_per_channel) != 0:
                for unit_idx in units_per_channel: 
                    valid_units.append(unit_idx)
        regional = []
        for unit in valid_units:
            regional.append(sorted[sorted[:,1] == unit])
        regional = np.vstack(regional)
        return regional
    else:
        trials = []
        for i in range(stimulus_onset_times.shape[0]):
            mask = np.logical_and(sorted[:,0]*samp_freq >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,   
                                 sorted[:,0]*samp_freq <= stimulus_onset_times[i]*samp_freq+samp_freq ) 
            trial = sorted[mask,:]
            trial[:,0] = (trial[:,0] - trial[:,0].min()) 
            trials.append(trial)
        return spikes_times, spikes_clusters, spikes_amps, spikes_depths, sorted, trials


def load_kilosort_good_ibl_units(rootpath, sub_id, roi='all', keep_active_trials = True, samp_freq=30_000):
    '''
    
    '''
    
    spikes_times = np.load(f'{rootpath}/{sub_id}/sorted/spikes_times.npy')
    spikes_clusters = np.load(f'{rootpath}/{sub_id}/sorted/spikes_clusters.npy')
    good_ibl_units = np.load(f'{rootpath}/{sub_id}/sorted/good_ibl_units.npy')
    stimulus_onset_times = np.load(f'{rootpath}/{sub_id}/misc/stimulus_onset_times.npy') # unit: seconds
    
    if keep_active_trials:
        active_trials_ids = np.load(f'{rootpath}/{sub_id}/behaviors/active_trials_ids.npy')
        stimulus_onset_times = stimulus_onset_times[active_trials_ids]
        
    spikes_indices = np.concatenate([spikes_times.reshape(-1,1), spikes_clusters.reshape(-1,1)], axis=1)
        
    if roi != 'all':
        clusters_channels = np.load(f'{rootpath}/{sub_id}/sorted/clusters_channels.npy', allow_pickle=True)
        channels_rois = np.load(f'{rootpath}/{sub_id}/sorted/channels_rois.npy', allow_pickle=True)
        channels_rois = np.vstack([np.arange(384), channels_rois]).transpose()
        valid_channels = channels_rois[[roi in x for x in channels_rois[:,-1]], 0]
        valid_channels = np.unique(valid_channels).astype(int)
        valid_units = []
        for c in valid_channels:
            units_per_channel = list(np.where(clusters_channels == c)[0])
            if len(units_per_channel) != 0:
                for unit_idx in units_per_channel: 
                    valid_units.append(unit_idx)
        good_regional_units = np.intersect1d(valid_units, good_ibl_units)
        good_sorted_data = np.vstack([spikes_indices[spikes_indices[:,1].astype(int) == unit] for unit in good_regional_units])
    else:
        good_sorted_data = np.vstack([spikes_indices[spikes_indices[:,1].astype(int) == unit] for unit in good_ibl_units])
        
    tmp = pd.DataFrame({'time': good_sorted_data[:,0], 'old_unit': good_sorted_data[:,1].astype(int)})
    tmp["old_unit"] = tmp["old_unit"].astype("category")
    tmp["new_unit"] = pd.factorize(tmp.old_unit)[0]
    good_sorted_indices = np.array(tmp)[:,[0,2]]
        
    return good_sorted_indices


def load_kilosort_unsorted_data(rootpath, sub_id, roi='all', keep_active_trials = True, samp_freq=30_000):
    '''
    
    '''    
    spikes_times = np.load(f'{rootpath}/{sub_id}/sorted/spikes_times.npy')
    spikes_clusters = np.load(f'{rootpath}/{sub_id}/sorted/spikes_clusters.npy')
    clusters_channels = np.load(f'{rootpath}/{sub_id}/sorted/clusters_channels.npy', allow_pickle=True)
    stimulus_onset_times = np.load(f'{rootpath}/{sub_id}/misc/stimulus_onset_times.npy') # unit: seconds
    
    if keep_active_trials:
        active_trials_ids = np.load(f'{rootpath}/{sub_id}/behaviors/active_trials_ids.npy')
        stimulus_onset_times = stimulus_onset_times[active_trials_ids]
    
    spikes_channels = np.array([clusters_channels[i] for i in spikes_clusters])
    unsorted = np.concatenate([spikes_times.reshape(-1,1), spikes_channels.reshape(-1,1)], axis=1)
    
    if roi != 'all':
        channels_rois = np.load(f'{rootpath}/{sub_id}/sorted/channels_rois.npy', allow_pickle=True)
        channels_rois = np.vstack([np.arange(384), channels_rois]).transpose()
        valid_channels = channels_rois[[roi in x for x in channels_rois[:,-1]], 0]
        valid_channels = np.unique(valid_channels).astype(int)
        regional = []
        for channel in valid_channels:
            regional.append(unsorted[unsorted[:,1] == channel])
        regional = np.vstack(regional)
        return regional
    else:
        return unsorted
    

def load_kilosort_localizations_data(rootpath, sub_id, roi='all', keep_active_trials=True, samp_freq=30_000):
    '''
    to do: load aligned spike indices not original
    '''
    # load aligned spike indices not original
    spikes_indices = np.load(f'{rootpath}/{sub_id}/sorted/localization_results/aligned_kilosort_spike_train.npy')   
    localization_features = np.load(f'{rootpath}/{sub_id}/sorted/localization_results/aligned_kilosort_localizations.npy')
    maxptp = np.load(f'{rootpath}/{sub_id}/sorted/localization_results/aligned_kilosort_maxptp.npy')
    clusters_channels = np.load(f'{rootpath}/{sub_id}/sorted/clusters_channels.npy', allow_pickle=True)
    np1_channel_map = np.load(f'{rootpath}/{sub_id}/misc/np1_channel_map.npy')
    stimulus_onset_times = np.load(f'{rootpath}/{sub_id}/misc/stimulus_onset_times.npy') # unit: seconds
    
    if keep_active_trials:
        active_trials_ids = np.load(f'{rootpath}/{sub_id}/behaviors/active_trials_ids.npy')
        stimulus_onset_times = stimulus_onset_times[active_trials_ids]
        
    spikes_channels = np.array([clusters_channels[i] for i in spikes_indices[:,1]]).reshape(-1,1)
    unsorted = np.concatenate([spikes_indices, spikes_channels, 
                               localization_features[:,[0,3]], maxptp.reshape(-1,1)], axis=1)
    
    # remove bad small spikes that get localized on the boundaries  
    mask = np.logical_and(unsorted[:,3] > -85, unsorted[:,3] < 158)
    unsorted = unsorted[mask]
        
    if roi != 'all':
        clusters_channels = np.load(f'{rootpath}/{sub_id}/sorted/clusters_channels.npy', allow_pickle=True)
        channels_rois = np.load(f'{rootpath}/{sub_id}/sorted/channels_rois.npy', allow_pickle=True)
        channels_rois = np.vstack([np.arange(384), channels_rois]).transpose()
        valid_channels = channels_rois[[roi in x for x in channels_rois[:,-1]], 0]
        valid_channels = np.unique(valid_channels).astype(int)
        print(f'found {len(valid_channels)} channels in roi {roi}')
        
        # add z_reg later 
        regional = []
        for i in valid_channels:
            regional.append(unsorted[unsorted[:,2] == i])
        regional = np.vstack(regional)
        trials = []
        for i in range(stimulus_onset_times.shape[0]):
            mask = np.logical_and(regional[:,0] >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,   
                             regional[:,0] <= stimulus_onset_times[i]*samp_freq+samp_freq ) 
            trial = regional[mask,:]
            trials.append(trial[:,[0,3,4,5]]) 
        return trials
    else:
        trials = []
        for i in range(stimulus_onset_times.shape[0]):
            mask = np.logical_and(unsorted[:,0] >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,   
                                 unsorted[:,0] <= stimulus_onset_times[i]*samp_freq+samp_freq )    # 1.5 secs / trial
            trial = unsorted[mask,:]
            trials.append(trial)
        return unsorted, trials
    
    
def load_behaviors_data(rootpath, sub_id):
    '''
    
    '''
    behave_dict = np.load(f'{rootpath}/{sub_id}/behaviors/processed_behaviors.npy')
    behave_idx_dict = np.load(f'{rootpath}/{sub_id}/behaviors/behave_idx_dict.npy', allow_pickle=True)
    return behave_dict, behave_idx_dict
    

def load_kilosort_template_feature_mads(rootpath):
    '''
    
    '''    
    temp_amps = np.load(f'{rootpath}/kilosort_template/ks_template_amps.npy')
    x_mad_scaled = np.load(f'{rootpath}/kilosort_template/x_mad_scaled.npy')
    z_mad_scaled = np.load(f'{rootpath}/kilosort_template/z_mad_scaled.npy')
    return temp_amps, x_mad_scaled, z_mad_scaled
    
def preprocess_static_behaviors(behave_dict, keep_active_trials=True):
    '''
    extract choices, stimuli, rewards and priors.
    to do: use 'behave_idx_dict' to select behaviors instead of hard-coding.
    '''
    
    if keep_active_trials:
        choices = behave_dict[:,:,:,23:25].sum(2)[0,:,:]
        stimuli = behave_dict[:,:,:,19:21].sum(2)[0,:,:]
        rewards = behave_dict[:,:,:,25:27].sum(2)[0,:,:]
        priors = behave_dict[0,:,0,28:29]
    else:
        choices = behave_dict[:,:,:,22:24].sum(2)[0,:,:]
        stimuli = behave_dict[:,:,:,19:21].sum(2)[0,:,:]
        rewards = behave_dict[:,:,:,24:26].sum(2)[0,:,:]
        priors = behave_dict[0,:,0,27:28]     
    print('choices left: %.3f, right: %.3f'%((choices.sum(0)[0]/choices.shape[0]), (choices.sum(0)[1]/choices.shape[0])))
    print('stimuli left: %.3f, right: %.3f'%((np.sum(stimuli.argmax(1)==1)/stimuli.shape[0]), \
                                   (np.sum(stimuli.argmax(1)==0)/stimuli.shape[0])))
    print('reward wrong: %.3f, correct: %.3f'%((rewards.sum(0)[0]/rewards.shape[0]), (rewards.sum(0)[1]/rewards.shape[0])))
    
    
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
    
    
    
def compute_time_binned_neural_activity(data, data_type, stimulus_onset_times, regional=False, n_time_bins=30, samp_freq=30_000):
    '''
    for gmm, unsorted, sorted.
    '''
    binning = np.arange(0, 1.5, step=(1.5 - 0)/n_time_bins)
    n_trials = stimulus_onset_times.shape[0]
    neural_data = []
    
    if data_type=='clusterless':
        spikes_times, spikes_labels, spikes_probs = data
        n_gaussians = len(np.unique(spikes_labels))
        spikes_probs = spikes_probs[:, np.unique(spikes_labels)]
        trials = np.hstack([spikes_times.reshape(-1,1), spikes_labels.reshape(-1,1), spikes_probs])

        for i in range(n_trials):
            mask = np.logical_and(trials[:,0] >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,
                                  trials[:,0] <= stimulus_onset_times[i]*samp_freq+samp_freq
                                 )
            trial = trials[mask,:]
            trial[:,0] = (trial[:,0] - trial[:,0].min()) / samp_freq
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
    
    elif data_type=='unsorted':
        if regional:
            n_channels = len(np.unique(data[:,1]))
            tmp = pd.DataFrame({'time': data[:, 0], 'old_channel': data[:,1].astype(int)})
            tmp["old_channel"] = tmp["old_channel"].astype("category")
            tmp["new_channel"] = pd.factorize(tmp.old_channel)[0]
            data = np.array(tmp)[:,[0,2]] 
        else:
            n_channels = 384
            
        spikes_indices = data.copy() / samp_freq
        for i in range(n_trials):
            mask = np.logical_and(spikes_indices[:,0]*samp_freq >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,
                                  spikes_indices[:,0]*samp_freq <= stimulus_onset_times[i]*samp_freq+samp_freq )
            trial = spikes_indices[mask,:]
            trial[:,0] = (trial[:,0] - trial[:,0].min()) 
            trial[:,1] = trial[:,1] * samp_freq
            channels = trial[:,1].astype(int)
            time_bins = np.digitize(trial[:,0], binning, right=False)-1
            spike_count = np.zeros([n_channels, n_time_bins])
            np.add.at(spike_count, (channels, time_bins), 1) 
            neural_data.append(spike_count)
        neural_data = np.array(neural_data)
        
    elif data_type=='sorted':
        if regional:
            n_neurons = len(np.unique(data[:,1]))
            tmp = pd.DataFrame({'time': data[:,0]*samp_freq, 'old_unit': data[:,1].astype(int)})
            tmp['old_unit'] = tmp['old_unit'].astype('category')
            tmp['new_unit'] = pd.factorize(tmp.old_unit)[0]
            spikes_indices = np.array(tmp)[:,[0,2]]
        else:
            spikes_times, spikes_clusters = data
            spikes_times = spikes_times * samp_freq
            n_neurons = len(np.unique(spikes_clusters))
            spikes_indices = np.concatenate([spikes_times.reshape(-1,1), spikes_clusters.reshape(-1,1)], axis=1)
        for i in range(n_trials):
            mask = np.logical_and(spikes_indices[:,0] >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,
                                  spikes_indices[:,0] <= stimulus_onset_times[i]*samp_freq+samp_freq )
            trial = spikes_indices[mask,:]
            trial[:,0] = (trial[:,0] - trial[:,0].min()) / samp_freq
            neurons = trial[:,1].astype(int)
            time_bins = np.digitize(trial[:,0], binning, right=False)-1
            spike_count = np.zeros([n_neurons, n_time_bins])
            np.add.at(spike_count, (neurons, time_bins), 1) 
            neural_data.append(spike_count)
        neural_data = np.array(neural_data)
    
    elif data_type=='good units':
        spikes_indices = data
        spikes_indices[:,0] = spikes_indices[:,0] * samp_freq
        n_neurons = len(np.unique(spikes_indices[:,1]))
        for i in range(n_trials):
            mask = np.logical_and(spikes_indices[:,0] >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,
                                  spikes_indices[:,0] <= stimulus_onset_times[i]*samp_freq+samp_freq )
            trial = spikes_indices[mask,:]
            trial[:,0] = (trial[:,0] - trial[:,0].min()) / samp_freq
            neurons = trial[:,1].astype(int)
            time_bins = np.digitize(trial[:,0], binning, right=False)-1
            spike_count = np.zeros([n_neurons, n_time_bins])
            np.add.at(spike_count, (neurons, time_bins), 1) 
            neural_data.append(spike_count)
        neural_data = np.array(neural_data)
        
    elif data_type=='kilosort unsorted':
        if regional:
            n_channels = len(np.unique(data[:,1]))
            tmp = pd.DataFrame({'time': data[:,0]*samp_freq, 'old_unit': data[:,1].astype(int)})
            tmp['old_unit'] = tmp['old_unit'].astype('category')
            tmp['new_unit'] = pd.factorize(tmp.old_unit)[0]
            spikes_indices = np.array(tmp)[:,[0,2]]
        else:
            n_channels = 384 
            spikes_indices = data
            spikes_indices[:,0] = spikes_indices[:,0] * samp_freq
            
        for i in range(n_trials):
            mask = np.logical_and(spikes_indices[:,0] >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,
                                  spikes_indices[:,0] <= stimulus_onset_times[i]*samp_freq+samp_freq )
            trial = spikes_indices[mask,:]
            trial[:,0] = (trial[:,0] - trial[:,0].min()) / samp_freq
            channels = trial[:,1].astype(int)
            time_bins = np.digitize(trial[:,0], binning, right=False)-1
            spike_count = np.zeros([n_channels, n_time_bins])
            np.add.at(spike_count, (channels, time_bins), 1) 
            neural_data.append(spike_count)
        neural_data = np.array(neural_data)
    
    return neural_data