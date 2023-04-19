import numpy as np
from one.api import ONE
import brainbox.behavior.dlc as dlc
import brainbox.behavior.wheel as wh

"""
Code for preprocessing behavioral data from IBL.
Code adapted from https://github.com/int-brain-lab/paper-reproducible-ephys.
"""

def bin_spikes(spike_times, align_times, pre_time, post_time, bin_size, weights=None):

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
    bin_vals, t = bin_spikes(times, events, pre_time, post_time, bin_size, weights=weights)
    bin_count, _ = bin_spikes(times, events, pre_time, post_time, bin_size)
    bin_count[bin_count == 0] = 1
    bin_vals = bin_vals / bin_count

    return bin_vals, t


def featurize_behavior(eid, t_before=0.5, t_after=1.0, bin_size=0.05):
    """
    
    """
    
    one = ONE(base_url = 'https://alyx.internationalbrainlab.org', silent=True)
    
    # load trials
    trials = one.load_object(eid, 'trials')
    trial_idx = np.arange(trials['stimOn_times'].shape[0])

    # filter out trials with no choice
    choice_filter = np.where(trials['choice'] != 0)
    trials = {key: trials[key][choice_filter] for key in trials.keys()}
    trial_idx = trial_idx[choice_filter]

    # filter out trials with no contrast
    contrast_filter = ~np.logical_or(trials['contrastLeft'] == 0, trials['contrastRight'] == 0)
    trials = {key: trials[key][contrast_filter] for key in trials.keys()}
    trial_idx = trial_idx[contrast_filter]
    nan_idx = np.c_[np.isnan(trials['stimOn_times']), 
                    np.isnan(trials['firstMovement_times']), 
                    np.isnan(trials['goCue_times']),
                    np.isnan(trials['response_times']), 
                    np.isnan(trials['feedback_times']),
                    np.isnan(trials['stimOff_times'])]
    kept_idx = np.sum(nan_idx, axis=1) == 0

    trials = {key: trials[key][kept_idx] for key in trials.keys()}
    trial_idx = trial_idx[kept_idx]
    
    
    # select active trials
    t_before = 0.5
    t_after = 1.0
    bin_size = 0.05

    ref_event = trials['firstMovement_times']
    diff1 = ref_event - trials['stimOn_times']
    diff2 = trials['feedback_times'] - ref_event
    t_select1 = np.logical_and(diff1 > 0.0, diff1 < t_before - 0.1)
    t_select2 = np.logical_and(diff2 > 0.0, diff2 < t_after - 0.1)
    t_select = np.logical_and(t_select1, t_select2)

    trials = {key: trials[key][t_select] for key in trials.keys()}
    trial_idx = trial_idx[t_select]
    ref_event = ref_event[t_select]

    n_active_trials = ref_event.shape[0]

    n_trials = n_active_trials
    print('number of trials found: {} (active: {})'.format(n_trials, n_active_trials))
    
    # load in dlc
    left_dlc = one.load_object(eid, 'leftCamera', 
                               attribute=['dlc', 'features', 'times', 'ROIMotionEnergy'], collection='alf')
    assert (left_dlc['times'].shape[0] == left_dlc['dlc'].shape[0])
    left_dlc['dlc'] = dlc.likelihood_threshold(left_dlc['dlc'], threshold=0)
    
    # get right paw speed (closer to camera)
    paw_speed = dlc.get_speed(left_dlc['dlc'], left_dlc['times'], camera='left', feature='paw_r')

    # get pupil diameter
    if 'features' in left_dlc.keys():
        pupil_diameter = left_dlc.pop('features')['pupilDiameter_smooth']
        if np.sum(np.isnan(pupil_diameter)) > 0:
            pupil_diameter = dlc.get_smooth_pupil_diameter(dlc.get_pupil_diameter(left_dlc['dlc']), 'left')
    else:
        pupil_diameter = dlc.get_smooth_pupil_diameter(dlc.get_pupil_diameter(left_dlc['dlc']), 'left')

    # get wheel velocity
    wheel = one.load_object(eid, 'wheel')
    vel = wh.velocity(wheel['timestamps'], wheel['position'])
    wheel_timestamps = wheel['timestamps'][~np.isnan(vel)]
    vel = vel[~np.isnan(vel)]
    
    # time binning
    n_tbins = int((t_after + t_before) / bin_size)
    # choice
    choice = (trials['choice'] > 0 ).astype(int) 
    # wheel velocity
    bin_vel, _ = bin_norm(wheel_timestamps, ref_event, t_before, t_after, bin_size, weights=vel)
    # left motion energy
    bin_left_me, _ = bin_norm(left_dlc['times'], ref_event, t_before, t_after, bin_size, 
                              weights=left_dlc['ROIMotionEnergy'])
    # paw speed
    bin_paw_speed, _ = bin_norm(left_dlc['times'], ref_event, t_before, t_after, bin_size, weights=paw_speed)
    # pupil diameter
    bin_pup_dia, _ = bin_norm(left_dlc['times'], ref_event, t_before, t_after, bin_size, weights=pupil_diameter)
    
    return choice, bin_left_me, bin_vel, bin_paw_speed, bin_pup_dia

