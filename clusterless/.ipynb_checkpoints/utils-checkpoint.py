import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas

from sklearn.mixture import GaussianMixture

import isosplit

class NP1DataLoader():
    def __init__(self, 
                 probe_id, geom_path, ephys_path, behavior_path):
        '''
        
        '''
        
        self.pid = probe_id
        self.geom = np.load(geom_path)
        self.ephys_path = ephys_path
        self.behavior_path = behavior_path
        
        # load spike sorting loader
        self.one = ONE(base_url = 'https://openalyx.internationalbrainlab.org', 
                       password = 'international', silent = True)
        ba = AllenAtlas()
        self.eid, probe = self.one.pid2eid(self.pid)
        self.sl = SpikeSortingLoader(pid = self.pid, one = self.one, atlas = ba)
        self.spikes, self.clusters, self.channels = self.sl.load_spike_sorting()
        self.clusters = self.sl.merge_clusters(self.spikes, self.clusters, self.channels)
        print(f'Session ID: {self.eid}')
        print(f'Probe ID: {self.pid} ({probe})')
        
        
        # load meta data
        stim_on_times = self.one.load_object(self.eid, 'trials', collection='alf')['stimOn_times']
        active_trials = np.load(f'{behavior_path}/{self.eid}_trials.npy')
        self.stim_on_times = stim_on_times[active_trials]
        self.n_trials = self.stim_on_times.shape[0]
        print(f'First trial stimulus onset time: {self.stim_on_times[0]:.2f} sec')
        print(f'Last trial stimulus onset time: {self.stim_on_times[-1]:.2f} sec') 
        
        
    def _partition_brain_regions(self, data, region, clusters_or_channels, 
                                 partition_type = None, good_units = []):
        '''
        
        '''
        region_of_interest = clusters_or_channels['acronym']
        region_of_interest = np.c_[np.arange(region_of_interest.shape[0]), region_of_interest]
        region_of_interest = region_of_interest[[region in x.lower() for x in region_of_interest[:,-1]], 0]
        region_of_interest = np.unique(region_of_interest).astype(int)
        if len(good_units) != 0:
            region_of_interest = np.intersect1d(region_of_interest, good_units)
        print(f'Found {len(region_of_interest)} {partition_type} in region {region}')
        regional = []
        for roi in region_of_interest:
            regional.append(data[data[:,1] == roi])
        return np.vstack(regional)
        
    
    def _partition_into_trials(self, data):
        '''
        
        '''
        trials = []
        for i in range(self.n_trials):
            mask = np.logical_and(data[:,0] >= self.stim_on_times[i] - 0.5,   
                                  data[:,0] <= self.stim_on_times[i] + 1.0)
            trials.append(data[mask])
        return trials
        
        
    def load_all_units(self, region = 'all'):
        '''
        
        '''
        spike_channels = [self.clusters.channels[i] for i in self.spikes.clusters]
        sorted = np.c_[self.spikes.times, self.spikes.clusters]
        
        if region != 'all':
            sorted = self._partition_brain_regions(sorted, region, self.clusters, 'KS units')
            
        return self._partition_into_trials(sorted)
        
    
    def load_good_units(self, region = 'all'):
        '''
        
        '''
        spike_channels = [self.clusters.channels[i] for i in self.spikes.clusters]
        good_units = self.clusters['cluster_id'][self.clusters['label'] == 1]
        sorted = np.c_[self.spikes.times, self.spikes.clusters]
        
        if region != 'all':
            sorted = self._partition_brain_regions(sorted, region, self.clusters, 'good units', good_units)
        else:
            print(f'Found {len(good_units)} good KS units')
        
        tmp = pd.DataFrame({'spike_time': sorted[:,0], 'old_unit': sorted[:,1].astype(int)})
        tmp["old_unit"] = tmp["old_unit"].astype("category")
        tmp["new_unit"] = pd.factorize(tmp.old_unit)[0]
        sorted = np.array(tmp[['spike_time','new_unit']])
        
        return self._partition_into_trials(sorted)
    
    
    def load_thresholded_units(self, region = 'all'):
        '''
        
        '''
        spike_index = np.load(f'{self.ephys_path}/spike_index.npy') 
        spike_times, spike_channels = spike_index.T
        spike_times = self.sl.samples2times(spike_times)
        unsorted = np.c_[spike_times, spike_channels]  
        
        if region != 'all':
            unsorted = self._partition_brain_regions(unsorted, region, self.channels, 'channels')
            
        return self._partition_into_trials(unsorted)
    
    
    def load_spike_features(self, region = 'all'):
        '''
        
        '''
        spike_index = np.load(f'{self.ephys_path}/spike_index.npy') 
        localization_results = np.load(f'{self.ephys_path}/localization_results.npy')
        spike_times, spike_channels = spike_index.T
        spike_times = self.sl.samples2times(spike_times)
        unsorted = np.c_[spike_times, spike_channels, localization_results]  
        
        if region != 'all':
            unsorted = self._partition_brain_regions(unsorted, region, self.channels, 'channels')
            
        return self._partition_into_trials(unsorted)
    
    
    def relocalize_kilosort(self, data_path, region = 'all'):
        '''
        
        '''
        spike_train = np.load(f'{data_path}/aligned_spike_train.npy')
        spike_index = np.load(f'{data_path}/aligned_spike_index.npy') 
        localization_results = np.load(f'{data_path}/aligned_localizations.npy')
        maxptp = np.load(f'{data_path}/aligned_maxptp.npy')             
        x, _, _, z, _ = localization_results.T
        spike_times, spike_channels = spike_index.T
        spike_times = self.sl.samples2times(spike_times)
        unsorted = np.c_[spike_times, spike_channels, x, z, maxptp]  
        
        if region != 'all':
            unsorted = self._partition_brain_regions(unsorted, region, self.channels, 'channels')
            
        return self._partition_into_trials(unsorted)
            
    
    def load_behaviors(self, behavior_type = 'static'):
        '''
        TO DO: Use dict to select behavior instead of hard coding.
        '''
        behave_dict = np.load(f'{self.behavior_path}/{self.eid}_feature.npy')
        
        if behavior_type == 'static':
            choices = behave_dict[:,:,:,23:25].sum(2)[0,:,:]
            stimuli = behave_dict[:,:,:,19:21].sum(2)[0,:,:]
            rewards = behave_dict[:,:,:,25:27].sum(2)[0,:,:]
            priors = behave_dict[0,:,0,28:29]

            print('Choices left: %.3f, right: %.3f'%((choices.sum(0)[0]/choices.shape[0]), 
                                                     (choices.sum(0)[1]/choices.shape[0])))
            print('Stimuli left: %.3f, right: %.3f'%((np.sum(stimuli.argmax(1)==1)/stimuli.shape[0]), 
                                           (np.sum(stimuli.argmax(1)==0)/stimuli.shape[0])))
            print('Reward wrong: %.3f, correct: %.3f'%((rewards.sum(0)[0]/rewards.shape[0]), 
                                                       (rewards.sum(0)[1]/rewards.shape[0])))

            # transform stimulus for plotting
            transformed_stimuli = []
            for s in stimuli:
                if s.argmax()==1:
                    transformed_stimuli.append(-1*s.sum())
                else:
                    transformed_stimuli.append(s.sum())
            transformed_stimuli = np.array(transformed_stimuli)

            # convert stimulus to categeorical variable for decoding
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(transformed_stimuli.reshape(-1,1))
            one_hot_stimuli = enc.transform(transformed_stimuli.reshape(-1,1)).toarray()
            return choices, stimuli, transformed_stimuli, one_hot_stimuli, enc.categories_, rewards, priors
            
        elif behavior_type == 'dynamic':
            motion_energy = behave_dict[0,:,:,18]
            wheel_velocity = behave_dict[0,:,:,27]
            wheel_speed = np.abs(wheel_velocity)
            paw_speed = behave_dict[0,:,:,15]
            nose_speed = behave_dict[0,:,:,16]
            pupil_diameter = behave_dict[0,:,:,17]
            return motion_energy, wheel_velocity, wheel_speed, paw_speed, nose_speed, pupil_diameter
        else:
            print('Invalid behavior type. Must be either static or dynamic.')
            
            
    def inverse_transform_stimulus(self, transformed_stimuli, enc_categories):
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
    
    
    def prepare_decoder_input(self, data, is_gmm = False, n_t_bins = 30, regional = False):
        '''

        '''
        t_binning = np.arange(0, 1.5, step=(1.5 - 0)/n_t_bins)
        
        decoder_input = []
        if is_gmm:
            spike_times, spike_labels = data[:,:2].T
            spike_probs = data[:,2:]
            n_gaussians = len(np.unique(spike_labels))
            # spike_probs = spike_probs[:, np.unique(spike_labels)]
            spike_train = np.c_[spike_times, spike_labels, spike_probs]

            for i in range(self.n_trials):
                mask = np.logical_and(spike_train[:,0] >= self.stim_on_times[i] - 0.5,
                                      spike_train[:,0] <= self.stim_on_times[i] + 1.0)
                trial = spike_train[mask]
                trial[:,0] = trial[:,0] - trial[:,0].min()
                t_bins = np.digitize(trial[:,0], t_binning, right = False)-1
                t_bins_lst = []
                for t in range(n_t_bins):
                    t_bin = trial[t_bins == t, 2:]
                    gmm_weights_lst = np.zeros(n_gaussians)
                    for k in range(n_gaussians):
                        gmm_weights_lst[k] = np.sum(t_bin[:,k])
                    t_bins_lst.append(gmm_weights_lst)
                decoder_input.append(np.array(t_bins_lst))
            decoder_input = np.array(decoder_input).transpose(0,2,1)
        else:
            spike_times, spike_units = data.T
            spike_train = np.c_[spike_times, spike_units]
            if regional:
                n_units = len(np.unique(spike_units))
                tmp = pd.DataFrame({'spike_time': spike_times, 'old_unit': spike_units.astype(int)})
                tmp['old_unit'] = tmp['old_unit'].astype("category")
                tmp['new_unit'] = pd.factorize(tmp.old_unit)[0]
                spike_train = np.array(tmp[['spike_time','new_unit']])
            else:
                n_units = spike_units.max().astype(int)+1
            for i in range(self.n_trials):
                mask = np.logical_and(spike_train[:,0] >= self.stim_on_times[i] - 0.5,
                                      spike_train[:,0] <= self.stim_on_times[i] + 1.0)
                trial = spike_train[mask]
                trial[:,0] = trial[:,0] - trial[:,0].min()
                units = trial[:,1].astype(int)
                t_bins = np.digitize(trial[:,0], t_binning, right = False)-1
                spike_count = np.zeros([n_units, n_t_bins])
                np.add.at(spike_count, (units, t_bins), 1) 
                decoder_input.append(spike_count)
            decoder_input = np.array(decoder_input)
        return decoder_input
    
    
    
class ADVIDataLoader():
    '''
    
    '''
    def __init__(self, data, behavior, n_t_bins = 30):
        self.data = data
        self.behavior = behavior
        self.n_trials = len(data)
        self.n_t_bins = n_t_bins
        self.t_binning = np.arange(0, 1.5, step = (1.5 - 0) / n_t_bins)
        
        # generate trial and time indices needed for model training
        self.trials = []; self.trial_ids = []; self.time_ids = []
        for k in range(self.n_trials):
            trial = data[k].copy()
            trial[:,0] = trial[:,0] - trial[:,0].min()
            t_bins = np.digitize(trial[:,0], self.t_binning, right = False) - 1
            t_bin_lst = []
            for t in range(self.n_t_bins):
                t_bin = trial[t_bins == t,1:]
                self.trial_ids.append(np.ones_like(t_bin[:,0]) * k)
                self.time_ids.append(np.ones_like(t_bin[:,0]) * t)
                t_bin_lst.append(t_bin)
            self.trials.append(t_bin_lst)
    
    
    def split_train_test(self, train_ids, test_ids):
        '''
        
        '''
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.train_behavior = self.behavior[train_ids]
        self.test_behavior = self.behavior[test_ids]
        
        trial_ids = np.concatenate(self.trial_ids)
        time_ids = np.concatenate(self.time_ids)
        trials = np.concatenate(np.concatenate(self.trials))

        train_mask = np.sum([trial_ids == idx for idx in train_ids], axis=0).astype(bool)
        test_mask = np.sum([trial_ids == idx for idx in test_ids], axis=0).astype(bool)
        train_trial_ids, test_trial_ids = trial_ids[train_mask], trial_ids[test_mask]
        train_time_ids, test_time_ids = time_ids[train_mask], time_ids[test_mask]
        train_trials, test_trials = trials[train_mask], trials[test_mask]
        
        return train_trials, train_trial_ids, train_time_ids, test_trials, test_trial_ids, test_time_ids
    
    def compute_lambda(self, gmm):
        C = len(gmm.means_)
        lambdas = []
        for k in self.train_ids:
            lam_lst = []
            for t in range(self.n_t_bins):
                lam = np.zeros((C, 2))
                t_bin = self.trials[k][t][:,1:]
                if len(t_bin) > 0:
                    cluster_ids = gmm.predict(t_bin)
                    for j in range(C):
                        if self.behavior[k] == 0:
                            lam[j, 0] = np.sum(cluster_ids == j)
                        else:
                            lam[j, 1] = np.sum(cluster_ids == j)
                lam_lst.append(lam)
            lambdas.append(lam_lst)
        n_left, n_right = np.sum(self.train_behavior == 0), np.sum(self.train_behavior == 1)
        p = n_right / (n_right + n_left)
        lambdas = ( np.array(lambdas).sum(0) / np.array([n_left, n_right]) ).transpose(1,0,2)
        return lambdas, p 
    
    
def fit_initial_gmm(spike_features):
    '''
    
    '''
    sub_weights = []; sub_means = []; sub_covs = []
    for channel in np.unique(spike_features[:,0]):
        sub_features = spike_features[spike_features[:,0] == channel, 1:]

        if sub_features.shape[0] > 10:
            try:
                isosplit_labels = isosplit.isosplit(sub_features.T, 
                                                    K_init = 20, 
                                                    min_cluster_size = 10,
                                                    whiten_cluster_pairs = 1, 
                                                    refine_clusters = 1)
            except AssertionError:
                continue
            except ValueError:
                continue
        elif sub_features.shape[0] < 2:
            continue
        else:
            isosplit_labels = np.zeros_like(sub_features[:,0])

        n_splits = np.unique(isosplit_labels).shape[0]
        # print(f'Channel {int(channel)} is split into {n_splits} modes')

        for label in np.arange(n_splits):
            mask = isosplit_labels == label
            sub_gmm = GaussianMixture(n_components=1, covariance_type='full',
                              init_params='k-means++')
            sub_gmm.fit(sub_features[mask])
            sub_labels = sub_gmm.predict(sub_features[mask])
            sub_weights.append(len(sub_labels)/len(spike_features))
            sub_means.append(sub_gmm.means_)
            sub_covs.append(sub_gmm.covariances_)

    gmm = GaussianMixture(n_components=len(np.hstack(sub_weights)), covariance_type='full', 
                          init_params='k-means++')
    gmm.weights_ = np.hstack(sub_weights)
    gmm.means_ = np.vstack(sub_means)
    gmm.covariances_ = np.vstack(sub_covs)
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(np.vstack(sub_covs)))
    
    return gmm

