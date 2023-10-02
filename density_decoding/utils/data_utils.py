"""Functions for loading and preprocessing data."""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas
import brainbox.behavior.dlc as dlc
import brainbox.behavior.wheel as wh

import isosplit
from sklearn.mixture import GaussianMixture


class BaseDataLoader():
    def __init__(
        self, 
        t_before,
        t_after,
        n_t_bins
    ):
        """
        A general data loader that works for all data types.
        
        Args:
            n_t_bins: number of time bins within each trial
        """
        self.t_before = t_before
        self.t_after = t_after
        self.trial_length = self.t_before + self.t_after
        self.n_t_bins = n_t_bins
        self.t_binning = np.arange(0, self.trial_length, step = self.trial_length/self.n_t_bins)
        self.type = "custom"
    
    
    def process_spike_features(
        self, 
        spike_times,
        spike_channels,
        spike_features, 
        trial_start_times,
        trial_end_times,
        valid_trials = None
    ):
        """
        Convert raw spike data into a suitable format for decoding.
        
        Args:
            spike_times: size (N,) array; in seconds
            spike_channels: size (N,) array
            spike_features: size (N, n_d) array, n_d = spike feature dim
            trial_start_times: (n_k,) array, n_k = number of trials (in seconds)
            trial_end_times: (n_k,) array (in seconds)
            valid_trials: trial index to keep
        
        Returns:
            bin_spike_features: a nested list w/ the structure:
                                for each k:
                                    for each t:
                                        size (n_t_k, 1+n_d) array, n_d = spike feature dim
            bin_trial_idxs: a list of trial index
            bin_time_idxs: a list of time bin index
        """
        
        n_trials = len(trial_start_times)
        
        if valid_trials is None:
            valid_trials = np.arange(n_trials) 
            
        spike_train = np.c_[spike_times, spike_channels, spike_features]
        
        bin_spike_features = []
        bin_trial_idxs, bin_time_idxs = [], []
        for k_idx in tqdm(range(len(valid_trials)), desc="Process spike features"):
            
            k = valid_trials[k_idx]
            mask = np.logical_and(
                spike_times >= trial_start_times[k],   
                spike_times <= trial_end_times[k]
            )
            sub_spike_train = spike_train[mask]
            sub_spike_train[:,0] = sub_spike_train[:,0] - sub_spike_train[:,0].min()
            t_bin_mask = np.digitize(sub_spike_train[:,0], self.t_binning, right=False).flatten() - 1
            
            spike_train_per_k = []
            for t in range(self.n_t_bins):
                spike_train_per_t_bin = sub_spike_train[t_bin_mask == t, 1:]
                spike_train_per_k.append(spike_train_per_t_bin)
                bin_trial_idxs.append(np.ones_like(spike_train_per_t_bin.T[0]) * k_idx)
                bin_time_idxs.append(np.ones_like(spike_train_per_t_bin.T[0]) * t)
            bin_spike_features.append(spike_train_per_k)
            
        return bin_spike_features, bin_trial_idxs, bin_time_idxs
    
    
    def process_behaviors(
        self, 
        time_points,
        raw_behaviors, 
        trial_start_times,
        trial_end_times,
    ):
        """
        Convert raw behavior traces into a suitable format for decoding.
        
        Args:
            time_points: size (n_time_points,) array (in seconds)
            raw_behaviors: size (n_time_points,) array
            trial_start_times: size (n_k,) array, n_k = number of trials (in seconds)
            trial_end_times: size (n_k,) array (in seconds)
        
        Returns:
            bin_behaviors: size (n_k, n_t) array
        """
        
        n_trials = len(trial_start_times)
        time_points = time_points.squeeze()
        raw_behaviors = raw_behaviors.squeeze()
        behavior_data = np.c_[time_points, raw_behaviors]
        
        bin_behaviors = np.zeros((n_trials, self.n_t_bins))
        for k in tqdm(range(n_trials), desc="Process behaviors"):
            
            mask = np.logical_and(
                time_points >= trial_start_times[k],   
                time_points <= trial_end_times[k]
            )
            sub_behaviors = behavior_data[mask]
            sub_behaviors[:,0] = sub_behaviors[:,0] - sub_behaviors[:,0].min()
            t_bin_mask = np.digitize(sub_behaviors[:,0], self.t_binning, right=False).flatten() - 1
            
            for t in range(self.n_t_bins):
                bin_behaviors[k, t] = sub_behaviors[t_bin_mask == t, 1].mean()
        
        return bin_behaviors
    
    
    def compute_spike_count_matrix(
        self, 
        spike_times, 
        spike_units, 
        trial_start_times,
        trial_end_times,
        valid_trials = None
    ):
        """
        Compute spike count matrix for spike-sorted and thresholded decoders.
        
        Args:
            spike_times: size (N,) array (in seconds)
            spike_units: size (N,) array (sorted / thresholded units)
            trial_start_times: size (n_k,) array, n_k = number of trials (in seconds)
            trial_end_times: size (n_k,) array (in seconds)
            valid_trials: trial index to keep
            
        Returns:
            spike_count_mat: size (n_k, n_c, n_t) array
        """
        
        n_trials = len(trial_start_times)
        valid_trials = np.arange(n_trials) if valid_trials is None else valid_trials
            
        spike_train = np.c_[spike_times, spike_units]

        spike_count_mat = []
        n_units = spike_units.max().astype(int) + 1
        for idx in tqdm(range(len(valid_trials)), desc="Compute spike count"):
            k = valid_trials[idx]
            
            mask = np.logical_and(
                spike_times >= trial_start_times[k],
                spike_times <= trial_end_times[k]
            )
            sub_spike_train = spike_train[mask]
            
            spike_count = np.zeros([n_units, self.n_t_bins])
            try:
                sub_spike_train[:,0] = sub_spike_train[:,0] - sub_spike_train[:,0].min()
                units = sub_spike_train[:,1].astype(int)
                t_bins = np.digitize(sub_spike_train[:,0], self.t_binning, right=False) - 1
                np.add.at(spike_count, (units, t_bins), 1) 
            except ValueError:
                print(f"no spikes in trial {k}.")
            spike_count_mat.append(spike_count)
            
        spike_count_mat = np.array(spike_count_mat)
        
        return spike_count_mat
    
    
    
class IBLDataLoader(BaseDataLoader):
    def __init__(
        self, 
        pid, 
        n_t_bins,
        base_url = "https://openalyx.internationalbrainlab.org",
        password = "international",
        prior_path=None,
        t_before = 0.5,
        t_after = 1.
    ):
        super().__init__(t_before, t_after, n_t_bins)
        """
        A data loader specific to IBL data.
        
        Args:
            pid: probe ID
            n_t_bins: number of time bins within each trial
        """
        self.pid = pid
        self.type = "ibl"
        
        # load spike sorting data from IBL
        ba = AllenAtlas()
        self.one = ONE(base_url=base_url, password=password, silent = True)
        self.eid, probe = self.one.pid2eid(self.pid)
        self.sl = SpikeSortingLoader(pid = self.pid, one = self.one, atlas = ba)
        self.spikes, self.clusters, self.channels = self.sl.load_spike_sorting()
        self.clusters = self.sl.merge_clusters(self.spikes, self.clusters, self.channels)
        
        print("pulling data from ibl database ..")
        print(f"eid: {self.eid}")
        print(f"pid: {self.pid}")
        
        # load meta data from IBL
        self.t_before, self.t_after = t_before, t_after
        self.trial_length = self.t_before + self.t_after
        self.bin_size = self.trial_length / n_t_bins
        stim_on_times = self.one.load_object(self.eid, "trials", collection="alf")["stimOn_times"] 
        self.behave_dict, valid_trials = self._featurize_behavior(prior_path=prior_path)
        self.stim_on_times = stim_on_times[valid_trials]
        self.n_trials = self.stim_on_times.shape[0]
        
        print(f"found {self.n_trials} trials from {self.stim_on_times[0]:.2f} to {self.stim_on_times[-1]:.2f} sec.")
        

    def check_available_brain_regions(self):
        """Check available brain regions for decoding."""
        print(np.unique(self.clusters["acronym"]))
        
        
    def load_spike_locations(self, spike_channels):
        """Get anatomical location (brain region) of each spike given its channel."""
        return self.channels["acronym"][spike_channels.astype(int)]

        
    def _partition_brain_regions(
        self, 
        data, 
        region, 
        partition_units, 
        partition_type=None, 
        good_units=[]
    ):
        """Partition data into different brain regions."""
        
        rois = partition_units["acronym"]
        rois = np.c_[np.arange(rois.shape[0]), rois]
        rois = rois[[region in x.lower() for x in rois[:,-1]], 0]
        rois = np.unique(rois).astype(int)
        
        if len(good_units) != 0:
            rois = np.intersect1d(rois, good_units)
        print(f"found {len(rois)} {partition_type} in region {region}")
        
        regional = []
        for idx in tqdm(range(len(rois)), desc="Partition brain regions"):
            roi = rois[idx]
            regional.append(data[data[:,1] == roi])
            
        return np.vstack(regional)
    
    
    def _partition_into_trials(self, data):
        """Partition data into different trials."""
        
        trials = []
        for i in range(self.n_trials):
            mask = np.logical_and(
                data[:,0] >= self.stim_on_times[i] - self.t_before,   
                data[:,0] <= self.stim_on_times[i] + self.t_after
            )
            trials.append(data[mask])
            
        return trials
    
    
    def load_all_sorted_units(self, region="all"):
        """
        Load all single units sorted by Kilosort 2.5. 
        
        Args:
            region: selected brain regions
            
        Returns:
            spike_count_mat: size (n_k, n_c, n_t) array
        """
        
        sorted = np.c_[self.spikes.times, self.spikes.clusters]
        
        is_regional = False
        if region != 'all':
            is_regional = True
            sorted = self._partition_brain_regions(
                sorted, region, self.clusters, "Kilosort units"
            )
            
        spike_times, spike_units = np.concatenate(
            self._partition_into_trials(sorted)
        ).T
    
        spike_count_mat = self.compute_spike_count_matrix(
            spike_times, 
            spike_units, 
            is_regional = is_regional
        )
        
        return spike_count_mat
        
    
    def load_good_sorted_units(self, region="all"):
        """
        Load single units that meet the quality control criteria. 
        
        Args:
            region: selected brain regions
            
        Returns:
            spike_count_mat: size (n_k, n_c, n_t) array
        """
        
        good_units = self.clusters['cluster_id'][self.clusters['label'] == 1]
        sorted = np.c_[self.spikes.times, self.spikes.clusters]
        
        is_regional = False
        if region != "all":
            is_regional = True
            sorted = self._partition_brain_regions(
                sorted, region, self.clusters, "good units", good_units
            )
        else:
            print(f"found {len(good_units)} good Kilosort units")
            good_sorted = []
            for good_unit in good_units:
                good_sorted.append(sorted[sorted[:,1] == good_unit])
            sorted = np.vstack(good_sorted)
        
        tmp = pd.DataFrame({"spike_time": sorted[:,0], "old_unit": sorted[:,1].astype(int)})
        tmp["old_unit"] = tmp["old_unit"].astype("category")
        tmp["new_unit"] = pd.factorize(tmp.old_unit)[0]
        sorted = np.array(tmp[["spike_time", "new_unit"]])
        
        spike_times, spike_units = np.concatenate(
            self._partition_into_trials(sorted)
        ).T
    
        spike_count_mat = self.compute_spike_count_matrix(
            spike_times, 
            spike_units, 
            is_regional = is_regional
        )
        
        return spike_count_mat
    
    
    def load_thresholded_units(self, spike_times, spike_channels, region="all"):
        """
        Load channels from multi-unit thresholding crossings.
        
        Args:
            spike_times: size (N,) array (in time samples)
            spike_channels: size (N,) array
            region: selected brain regions
            
        Returns:
            spike_count_mat: size (n_k, n_c, n_t) size
        """
        
        # convert spike times samples to seconds
        spike_times = self.sl.samples2times(spike_times)
        unsorted = np.c_[spike_times, spike_channels]  
        
        is_regional = False
        if region != 'all':
            is_regional = True
            unsorted = self._partition_brain_regions(
                unsorted, region, self.channels, "channels"
            )
            
        spike_times, spike_units = np.concatenate(
            self._partition_into_trials(unsorted)
        ).T
    
        spike_count_mat = self.compute_spike_count_matrix(
            spike_times, 
            spike_units, 
            is_regional = is_regional
        )
        
        return spike_count_mat
    
    
    def load_spike_features(
        self, 
        spike_times, 
        spike_channels, 
        spike_features, 
        region="all"
    ):
        """
        Convert raw spike data into a suitable format for decoding.
        
        Args:
            spike_times: size (N,) array (in time samples)
            spike_channels: size (N,) array
            spike_features: size (N, n_d) array, n_d = spike feature dim
            region: selected brain region
            
        Returns:
            bin_spike_features: a nested list w/ the structure:
                                for each k:
                                    for each t:
                                        size (n_t_k, 1+n_d) array, n_d = spike feature dim
        """
        
        spike_times = self.sl.samples2times(spike_times)
        unsorted = np.c_[spike_times, spike_channels, spike_features]  
        
        if region != 'all':
            unsorted = self._partition_brain_regions(
                unsorted, region, self.channels, "channels"
            )
        
        spike_times = unsorted[:,0]
        spike_channels = unsorted[:,1]
        spike_features = unsorted[:,2:]
        
        bin_spike_features, bin_trial_idxs, bin_time_idxs = self.process_spike_features(
            spike_times, 
            spike_channels, 
            spike_features,
            self.stim_on_times - self.t_before, 
            self.stim_on_times + self.t_after
        )
        
        return bin_spike_features, bin_trial_idxs, bin_time_idxs
    
    
    def compute_spike_count_matrix(
        self, 
        spike_times, 
        spike_units, 
        is_regional=False
    ):
        """
        Compute spike count matrix for spike-sorted and thresholded decoders.
        
        Args:
            spike_times: size (N,) array (in seconds)
            spike_units: size (N,) array (sorted / thresholded units)
            is_regional: whether from a brain region
            
        Returns:
            spike_count_mat: size (n_k, n_c, n_t) array
        """
        
        spike_train = np.c_[spike_times, spike_units]
        
        spike_count_mat = []
        if is_regional:
            n_units = len(np.unique(spike_units))
            tmp = pd.DataFrame({'spike_time': spike_times, 'old_unit': spike_units.astype(int)})
            tmp['old_unit'] = tmp['old_unit'].astype('category')
            tmp['new_unit'] = pd.factorize(tmp.old_unit)[0]
            spike_train = np.array(tmp[['spike_time','new_unit']])
        else:
            n_units = spike_units.max().astype(int)+1
            
        for k in tqdm(range(self.n_trials), desc="Compute spike count"):
            
            mask = np.logical_and(
                spike_train[:,0] >= self.stim_on_times[k] - self.t_before,
                spike_train[:,0] <= self.stim_on_times[k] + self.t_after
            )
            sub_spike_train = spike_train[mask]
            
            spike_count = np.zeros([n_units, self.n_t_bins])
            try:
                sub_spike_train[:,0] = sub_spike_train[:,0]-sub_spike_train[:,0].min()
                units = sub_spike_train[:,1].astype(int)
                t_bins = np.digitize(sub_spike_train[:,0], self.t_binning, right=False)-1
                np.add.at(spike_count, (units, t_bins), 1) 
            except ValueError:
                print(f'no spikes in trial {k}.')
            spike_count_mat.append(spike_count)
            
        spike_count_mat = np.array(spike_count_mat)
        
        return spike_count_mat
    
    
    def process_behaviors(self, behavior_type="choice"):
        """
        Convert raw behavior traces into a suitable format for decoding.
        
        Args:
            behavior_type: expected one of {"choice", "motion_energy", "wheel_velocity", "wheel_speed"}
        
        Returns:
            behaviors: size (n_k,) or (n_k, n_t) array for discrete or continuous variables
        """
        
        valid_types = ["choice", "prior", "contrast", "reward",
                       "motion_energy", "wheel_velocity", "wheel_speed",
                       "pupil_diameter", "paw_speed"]
        assert behavior_type in valid_types, f"invalid behavior type; expected one of {valid_types}."
        
        behaviors = self.behave_dict[behavior_type]
        return behaviors
    
    
    def _featurize_behavior(self, prior_path=None):
        """
        Preprocess behavioral data from IBL. Adapted from:
        https://github.com/int-brain-lab/paper-reproducible-ephys.
        """

        # load trials
        trials = self.one.load_object(self.eid, "trials")
        trial_idx = np.arange(trials["firstMovement_times"].shape[0])

        # filter out trials with no choice
        choice_filter = np.where(trials["choice"] != 0)
        trials = {key: trials[key][choice_filter] for key in trials.keys()}
        trial_idx = trial_idx[choice_filter]

        # filter out trials with no contrast
        contrast_filter = ~np.logical_or(trials["contrastLeft"] == 0, trials["contrastRight"] == 0)
        trials = {key: trials[key][contrast_filter] for key in trials.keys()}
        trial_idx = trial_idx[contrast_filter]
        nan_idx = np.c_[np.isnan(trials["stimOn_times"]), 
                        np.isnan(trials["firstMovement_times"]), 
                        np.isnan(trials["goCue_times"]),
                        np.isnan(trials["response_times"]), 
                        np.isnan(trials["feedback_times"]),
                        np.isnan(trials["stimOff_times"])]
        kept_idx = np.sum(nan_idx, axis=1) == 0

        trials = {key: trials[key][kept_idx] for key in trials.keys()}
        trial_idx = trial_idx[kept_idx]

        # select active trials
        ref_event = trials["firstMovement_times"] 
        # diff1 = ref_event - trials["stimOn_times"]
        # diff2 = trials["feedback_times"] - ref_event
        # t_select1 = np.logical_and(diff1 > 0.0, diff1 < self.t_before - 0.1)
        # t_select2 = np.logical_and(diff2 > 0.0, diff2 < self.t_after - 0.1)
        # t_select = np.logical_and(t_select1, t_select2)

        # trials = {key: trials[key][t_select] for key in trials.keys()}
        # trial_idx = trial_idx[t_select]
        # ref_event = ref_event[t_select]

        n_active_trials = ref_event.shape[0]

        n_trials = n_active_trials
        print("number of trials found: {} (active: {})".format(n_trials, n_active_trials))
        
        # load stimulus contrast 
        contrast = np.c_[trials['contrastLeft'], trials['contrastRight']]

        # load in dlc
        left_dlc = self.one.load_object(
            self.eid, "leftCamera", 
            attribute=["dlc", "features", "times", "ROIMotionEnergy"], 
            collection="alf"
        )
        assert (left_dlc["times"].shape[0] == left_dlc["dlc"].shape[0])
        left_dlc["dlc"] = dlc.likelihood_threshold(left_dlc["dlc"], threshold=0)

        # TO DO: the data quality of paw speed and pupil diameter is unreliable,
        #        switch to lightning-pose later.
        # get right paw speed (closer to camera)
        paw_speed = dlc.get_speed(left_dlc["dlc"], left_dlc["times"], camera="left", feature="paw_r")

        # get pupil diameter
        if "features" in left_dlc.keys():
            pupil_diameter = left_dlc.pop("features")["pupilDiameter_smooth"]
            if np.sum(np.isnan(pupil_diameter)) > 0:
                pupil_diameter = dlc.get_smooth_pupil_diameter(dlc.get_pupil_diameter(left_dlc["dlc"]), "left")
        else:
            pupil_diameter = dlc.get_smooth_pupil_diameter(dlc.get_pupil_diameter(left_dlc["dlc"]), "left")

        # get wheel velocity
        wheel = self.one.load_object(self.eid, "wheel")
        vel = wh.velocity(wheel["timestamps"], wheel["position"])
        wheel_timestamps = wheel["timestamps"][~np.isnan(vel)]
        vel = vel[~np.isnan(vel)]

        # time binning
        n_tbins = int(self.trial_length / self.bin_size)
        # choice
        choice = (trials["choice"] > 0 ).astype(int) 
        
        # reward
        reward = (trials["rewardVolume"] > 1) * 1.
        
        # wheel velocity
        bin_vel, _ = bin_norm(wheel_timestamps, ref_event, self.t_before, 
                              self.t_after, self.bin_size, weights=vel)
        # left motion energy
        bin_left_me, _ = bin_norm(left_dlc["times"], ref_event, self.t_before, 
                                  self.t_after, self.bin_size, 
                                  weights=left_dlc["ROIMotionEnergy"])
        # paw speed
        bin_paw_speed, _ = bin_norm(left_dlc["times"], ref_event, self.t_before, 
                                    self.t_after, self.bin_size, weights=paw_speed)
        # pupil diameter
        bin_pup_dia, _ = bin_norm(left_dlc["times"], ref_event, self.t_before, 
                                  self.t_after, self.bin_size, weights=pupil_diameter)
        
        behave_dict = {}
        behave_dict.update({"choice": choice})
        behave_dict.update({"contrast": contrast})
        behave_dict.update({"reward": reward})
        behave_dict.update({"motion_energy": bin_left_me})
        behave_dict.update({"wheel_velocity": bin_vel})
        behave_dict.update({"wheel_speed": np.abs(bin_vel)})
        behave_dict.update({"paw_speed": bin_paw_speed})
        behave_dict.update({"pupil_diameter": bin_pup_dia})
        
        # load priors
        try:
            prior = np.load(Path(prior_path) / f'prior_{self.eid}.npy')
            behave_dict.update({"prior": prior[trial_idx]})
        except:
            print("prior for this session is not found.")

        return behave_dict, trial_idx
    

def initilize_gaussian_mixtures(
    spike_features, 
    spike_channels=None, 
    method="isosplit", 
    n_c = 50,
    verbose=False
):
    """
    Fit a Gaussian mixture model to initialize the ADVI (CAVI) model. 

    Args:
        spike_channels: size (N,) array (for initial split) 
        spike_features: size (N, n_d) array, n_d = spike feature dim
        method: "isosplit" or "sklearn";
                If spike channels are provided, use isosplit to split 
                If spike channels not provided, use sklearn's Gaussian mixture model
        verbose: whether to print the splitting progress

    Returns:
        gmm: an object from sklearn.mixture.GaussianMixture().
    """
    valid_methods = ["isosplit", "sklearn"]
    assert method in valid_methods, f"invalid method; expected one of {valid_methods}."

    unique_chans = np.unique(spike_channels)
    
    if method == "sklearn":
        if spike_channels is not None:
            n_c = len(unique_chans)
        gmm = GaussianMixture(n_components = n_c).fit(spike_features)

    elif method == "isosplit":
        assert spike_channels is not None, "expected spike channels as input."

        n_spikes_required = 10
        min_n_spikes = 2
        
        subset_weights = []; subset_means = []; subset_covs = []
        for chan_idx in tqdm(range(len(unique_chans)), desc="Initialize GMM"):
            channel = unique_chans[chan_idx]
            
            subset_spike_features = spike_features[spike_channels == channel]

            if subset_spike_features.shape[0] > n_spikes_required:
                try:
                    spike_labels = isosplit.isosplit(
                        subset_spike_features.T, 
                        K_init = 20, 
                        min_cluster_size = 10,
                        whiten_cluster_pairs = 1, 
                        refine_clusters = 1
                    )
                except AssertionError:
                    continue
                except ValueError:
                    continue
            elif subset_spike_features.shape[0] < min_n_spikes:
                continue
            else:
                spike_labels = np.zeros_like(subset_spike_features.T[0])

            n_labels = np.unique(spike_labels).shape[0]
            if verbose:
                print(f'split channel {int(channel)} into {n_labels} components.')

            for label in np.arange(n_labels):
                mask = (spike_labels == label)
                subset_gmm = GaussianMixture(
                    n_components=1, 
                    covariance_type='full',
                    init_params='k-means++'
                )
                subset_gmm.fit(subset_spike_features[mask])
                subset_labels = subset_gmm.predict(subset_spike_features[mask])
                subset_weights.append(len(subset_labels)/len(spike_features))
                subset_means.append(subset_gmm.means_)
                subset_covs.append(subset_gmm.covariances_)

        # aggregate the subsets of Gaussian mixture models into one overall model
        n_c = len(np.hstack(subset_weights))
        gmm = GaussianMixture(n_components = n_c, 
                              covariance_type = 'full', 
                              init_params = 'k-means++')
        gmm.weights_ = np.hstack(subset_weights)
        gmm.means_ = np.vstack(subset_means)
        gmm.covariances_ = np.vstack(subset_covs)
        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))

    return gmm


def initialize_weight_matrix(gmm, spike_features):
    
    n_k = len(spike_features)
    n_t = len(spike_features[0])
    n_c = len(gmm.means_)
    
    weight_matrix = np.zeros((n_k, n_c, n_t))
    for k in tqdm(range(n_k), desc="Initialize weight matrix"):
        for t in range(n_t):
            if len(spike_features[k][t]) > 0:
                weight_matrix[k,:,t] = gmm.predict_proba(spike_features[k][t][:,1:]).sum(0)
    return weight_matrix


def bin_spikes(spike_times, align_times, pre_time, post_time, bin_size, weights=None):
    """
    Preprocess behavioral data from IBL. Adapted from:
    https://github.com/int-brain-lab/paper-reproducible-ephys.
    """
    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], n_bins))

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (np.floor((spike_times[ep[0]:ep[1]] - t[0]) / bin_size)).astype(np.int64)
        w = weights[ep[0]:ep[1]] if weights is not None else None
        r = np.bincount(xind, minlength=tscale.shape[0], weights=w)
        bins[i, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale


def bin_norm(times, events, pre_time, post_time, bin_size, weights):
    """
    Preprocess behavioral data from IBL. Adapted from:
    https://github.com/int-brain-lab/paper-reproducible-ephys.
    """
    bin_vals, t = bin_spikes(times, events, pre_time, post_time, bin_size, weights=weights)
    bin_count, _ = bin_spikes(times, events, pre_time, post_time, bin_size)
    bin_count[bin_count == 0] = 1
    bin_vals = bin_vals / bin_count

    return bin_vals, t


def sliding_window_behaviors(y, delta=5):
    n_k = len(y)

    y_pos = []
    y_window = []
    for k in range(n_k):
        window = [k-delta, k+delta+1] 
        if np.logical_and(window[0] >= 0, window[1] <= n_k):
            sub_y = y[window[0]:window[1]]
            y_pos.append(delta)
        elif window[0] < 0:
            sub_y = y[k:k+2*delta+1]
            y_pos.append(0)
        elif window[1] > n_k:
            sub_y = y[k-2*delta:k+1]
            y_pos.append(-1)
        y_window.append(sub_y)
    y_window = np.vstack(y_window)
    
    return y_window, y_pos