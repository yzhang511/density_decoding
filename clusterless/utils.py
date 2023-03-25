import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas



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
        print(f'First trial stimulus onset time: {self.stim_on_times[0]:.2f}')
        print(f'Last trial stimulus onset time: {self.stim_on_times[-1]:.2f}') 
        
        
    def _partition_brain_regions(self, data, region, clusters_or_channels, 
                                 partition_type = None, good_units = None):
        '''
        
        '''
        region_of_interest = clusters_or_channels['acronym']
        region_of_interest = np.c_[np.arange(region_of_interest.shape[0]), region_of_interest]
        region_of_interest = region_of_interest[[region in x.lower() for x in region_of_interest[:,-1]], 0]
        region_of_interest = np.unique(region_of_interest).astype(int)
        if len(good_units) != 0:
            region_of_interest = np.intersect1d(region_of_interest, good_units)
        print(f'Found {len(region_of_interest)} {partition_type} in region {region}.')
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
            print(f'Found {len(good_units)} good KS units.')
        
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
            
    
    def load_behaviors(self, ):
        '''
        
        '''
        
        pass
    
    
    def prepare_decoder_input(self, ):
        '''
        
        '''
        
        pass