import numpy as np

def load_unsorted_data(rootpath, sub_id, keep_active_trials=True, samp_freq=30_000):
    '''
    
    '''
    
    spikes_indices = np.load(f'{rootpath}/{sub_id}/unsorted/spikes_indices.npy')        # unit: time samples
    spikes_features = np.load(f'{rootpath}/{sub_id}/unsorted/spikes_features.npy')
    np1_channel_map = np.load(f'{rootpath}/{sub_id}/misc/np1_channel_map.npy')
    stimulus_onset_times = np.load(f'{rootpath}/{sub_id}/misc/stimulus_onset_times.npy') # unit: seconds
    
    if keep_active_trials:
        active_trials_ids = np.load(f'{rootpath}/{sub_id}/behaviors/active_trials_ids.npy')
        stimulus_onset_times = stimulus_onset_times[active_trials_ids]
        
    # add z_reg later 
    unsorted = np.concatenate([spikes_indices[:,0].reshape(-1,1), spikes_features[:,[0,2,4]]], axis=1)
    trials = []
    for i in range(stimulus_onset_times.shape[0]):
        mask = np.logical_and(unsorted[:,0] >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,   
                             unsorted[:,0] <= stimulus_onset_times[i]*samp_freq+samp_freq )        # 1.5 secs / trial
        trial = unsorted[mask,:]
        trial[:,0] = (trial[:,0] - trial[:,0].min()) / samp_freq
        trials.append(trial)
    
    return spikes_indices, spikes_features, np1_channel_map, stimulus_onset_times, unsorted, trials
    
    
    
def load_kilosort_sorted_data(rootpath, sub_id, keep_active_trials = True, samp_freq=30_000):
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
    trials = []
    for i in range(stimulus_onset_times.shape[0]):
        mask = np.logical_and(sorted[:,0]*samp_freq >= stimulus_onset_times[i]*samp_freq-samp_freq*0.5,   
                             sorted[:,0]*samp_freq <= stimulus_onset_times[i]*samp_freq+samp_freq ) 
        trial = sorted[mask,:]
        trial[:,0] = (trial[:,0] - trial[:,0].min()) 
        trials.append(trial)
    return spikes_times, spikes_clusters, spikes_amps, spikes_depths, sorted, trials


# to do: load_kilosort_good_ibl_units()

# to do: load_kilosort_unsorted_data()
    
    
def load_behaviors_data(rootpath, sub_id):
    '''
    
    '''
    behave_dict = np.load(f'{rootpath}/{sub_id}/behaviors/processed_behaviors.npy')
    behave_idx_dict = np.load(f'{rootpath}/{sub_id}/behaviors/behave_idx_dict.npy', allow_pickle=True)
    return behave_dict, behave_idx_dict
    

def load_kilosort_template_feature_mads(rootpath, sub_id):
    '''
    
    '''    
    temp_amps = np.load(f'{rootpath}/{sub_id}/misc/ks_template_amps.npy')
    x_mad_scaled = np.load(f'{rootpath}/{sub_id}/misc/x_mad_scaled.npy')
    z_mad_scaled = np.load(f'{rootpath}/{sub_id}/misc/z_mad_scaled.npy')
    return temp_amps, x_mad_scaled, z_mad_scaled
    
def preprocess_static_behaviors(behave_dict):
    '''
    extract choices, stimuli, rewards and priors.
    '''
    choices = behave_dict[:,:,:,22:24].sum(2)[0,:,:]
    print('choices left: %.3f, right: %.3f'%((choices.sum(0)[0]/choices.shape[0]), (choices.sum(0)[1]/choices.shape[0])))
    
    stimuli = behave_dict[:,:,:,19:21].sum(2)[0,:,:]
    print('stimuli left: %.3f, right: %.3f'%((np.sum(stimuli.argmax(1)==1)/stimuli.shape[0]), \
                                   (np.sum(stimuli.argmax(1)==0)/stimuli.shape[0])))
    
    rewards = behave_dict[:,:,:,24:26].sum(2)[0,:,:]
    print('reward wrong: %.3f, correct: %.3f'%((rewards.sum(0)[0]/rewards.shape[0]), (rewards.sum(0)[1]/rewards.shape[0])))
    
    priors = behave_dict[0,:,0,27:28]
    
    transformed_stimuli = []
    for s in stimuli:
        if s.argmax()==1:
            transformed_stimuli.append(-1*s.sum())
        else:
            transformed_stimuli.append(s.sum())
    transformed_stimuli = np.array(transformed_stimuli)

    return choices, stimuli, transformed_stimuli, rewards, priors
    
# to do: compute_time_binned_neural_activity() - for clusterless, unsorted, sorted. 