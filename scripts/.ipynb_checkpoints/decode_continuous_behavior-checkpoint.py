import numpy as np
import random
import torch

from sklearn.model_selection import KFold

from clusterless.utils import NP1DataLoader, ADVIDataLoader, fit_initial_gmm
from clusterless.encoder import ADVI
from clusterless.decoder import dynamic_decoder, sliding_window_decoder


seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.double)

session = 'DY_009'
region = 'po'
behavior = 'wheel_speed'
if region != 'all':
    is_regional = True
print(f'Decoding {behavior} using ephys data from {region} brain region in session {session}:')

np1_data_loader = NP1DataLoader(
    probe_id = 'febb430e-2d50-4f83-87a0-b5ffbb9a4943', 
    geom_path = f'/mnt/3TB/yizi/danlab/Subjects/{session}/np1_channel_map.npy', 
    ephys_path = f'/mnt/3TB/yizi/danlab/Subjects/{session}/subtraction_results_threshold_5', 
    behavior_path = '/mnt/3TB/yizi/Downloads/ONE/openalyx.internationalbrainlab.org/paper_repro_ephys_data/figure9_10/original_data'
)

trials = np1_data_loader.relocalize_kilosort(
    data_path = f'/mnt/3TB/yizi/danlab/Subjects/{session}/kilosort_localizations',
    region = region)

motion_energy, wheel_velocity, wheel_speed, paw_speed, nose_speed, pupil_diameter = \
    np1_data_loader.load_behaviors('dynamic')

if behavior == 'motion_energy':
    y = motion_energy
elif behavior == 'wheel_speed':
    y = wheel_speed

advi_data_loader = ADVIDataLoader(
                         data = trials, 
                         behavior = y, 
                         n_t_bins = 30)

gmm = fit_initial_gmm(np.concatenate(trials)[:,1:])

Nt = advi_data_loader.n_t_bins
Nc = gmm.means_.shape[0]
Nd = gmm.means_.shape[1]
print(f'Finished fitting the initial GMM with {Nc} components.')


kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for i, (train_ids, test_ids) in enumerate(kf.split(advi_data_loader.behavior)):
    
    print(f'Started decoding {i+1} / 5 folds:')
    res_metrics = {}
    
    train_trials, train_trial_ids, train_time_ids, \
    test_trials, test_trial_ids, test_time_ids = advi_data_loader.split_train_test(train_ids, test_ids)
    
    s = torch.tensor(train_trials[:,1:])
    y = torch.tensor(advi_data_loader.behavior)
    ks = torch.tensor(train_trial_ids)
    ts = torch.tensor(train_time_ids)

    Nk = len(advi_data_loader.train_ids)
    
    print(f'Started fitting ADVI:')
    
    batch_size = 1
    batch_ids = list(zip(*(iter(advi_data_loader.train_ids),) * batch_size))
    advi = ADVI(batch_size, Nt, Nc, Nd, gmm.means_, gmm.covariances_)
    optim = torch.optim.Adam(advi.parameters(), lr=1e-2)
    elbos = advi.train_advi(s, y, ks, ts, batch_ids, optim, max_iter=30)
    
    print(f'Finished fitting ADVI.')
    
    train = advi_data_loader.train_ids
    test = advi_data_loader.test_ids
    
    print(f'Started decoding using the vanilla GMM:')
    
    spike_train = np.concatenate(trials)
    spike_times = spike_train[:,0]
    spike_channels = spike_train[:,1]
    spike_labels = gmm.predict(spike_train[:,2:])
    spike_probs = gmm.predict_proba(spike_train[:,2:])

    vanilla_gmm = np1_data_loader.prepare_decoder_input(
        np.c_[spike_times, spike_labels, spike_probs],
        is_gmm = True, n_t_bins = 30, regional = is_regional)
    
    y_train, _, y_pred = dynamic_decoder(vanilla_gmm, advi_data_loader.behavior, train, test)
    _, _, _, r2, mse, corr = sliding_window_decoder(
        vanilla_gmm, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'gmm': [r2, mse, corr]})
    
    print(f'Started decoding using multi-unit thresholding:')
    
    thresholded = np1_data_loader.prepare_decoder_input(
        np.c_[spike_times, spike_channels],
        is_gmm = False, n_t_bins = 30, regional = is_regional)
    
    _, _, _, r2, mse, corr = sliding_window_decoder(
        thresholded, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'thresholded': [r2, mse, corr]})
    
    print(f'Started decoding using encoded GMM:')
    
    _, enc_all = advi.encode_gmm(advi_data_loader.trials, train, test, y_train, y_pred)
    _, _, _, r2, mse, corr = sliding_window_decoder(
        enc_all, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'enc_gmm': [r2, mse, corr]})
    
    
    print(f'Started decoding using all KS units:')
    
    ks_all = np1_data_loader.load_all_units(region=region)
    ks_all = np.concatenate(ks_all)

    all_units = np1_data_loader.prepare_decoder_input(
        ks_all, is_gmm = False, n_t_bins = 30, regional = is_regional)
    
    _, _, _, r2, mse, corr = sliding_window_decoder(
        all_units, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'all_units': [r2, mse, corr]})
    
    print(f'Started decoding using good KS units:')
    
    ks_good = np1_data_loader.load_good_units(region=region)
    ks_good = np.concatenate(ks_good)

    good_units = np1_data_loader.prepare_decoder_input(
        ks_good, is_gmm = False, n_t_bins = 30, regional = is_regional)
    
    _, _, _, r2, mse, corr = sliding_window_decoder(
        good_units, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'good_units': [r2, mse, corr]})
    
    np.save(f'../neurips_results/{session}/{region}/{behavior}/res_fold{i+1}.npy', res_metrics)
    
    
    
    