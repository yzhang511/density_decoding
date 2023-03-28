import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold

import torch

from clusterless.utils import NP1DataLoader, ADVIDataLoader, fit_initial_gmm
from clusterless.encoder import CAVI
from clusterless.decoder import static_decoder

seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.double)


session = 'DY_009'
region = 'po'
T = 15
behavior = 'choice'
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

# trials = np1_data_loader.load_spike_features(region = region)

choices, stimuli, transformed_stimuli, one_hot_stimuli, enc_categories_, rewards, priors = \
    np1_data_loader.load_behaviors('static')

advi_data_loader = ADVIDataLoader(
                         data = trials, 
                         behavior = choices.argmax(1), 
                         n_t_bins = T)

gmm = fit_initial_gmm(np.concatenate(trials)[:,1:])
init_C = len(gmm.means_) 
# gmm.fit(np.concatenate(trials)[:,2:])
# print(f'Finished fitting the initial GMM with {C} components.')

kf = KFold(n_splits=6, shuffle=True, random_state=seed)
kf_train_ids = []; kf_test_ids = []
for i, (train_ids, test_ids) in enumerate(kf.split(advi_data_loader.behavior)):
    kf_train_ids.append(train_ids)
    kf_test_ids.append(test_ids)
    
# choose C via validation
print('Start choosing C:') 

spike_train = np.concatenate(trials)
spike_times = spike_train[:,0]
spike_channels = spike_train[:,1]

candidates = list(range(10, init_C, init_C // 10))
val_accs = []
for c in candidates:
    gmm = GaussianMixture(n_components = c, 
                          covariance_type = 'full', 
                          init_params = 'k-means++')
    gmm.fit(spike_train[:,2:])
    
    spike_labels = gmm.predict(spike_train[:,2:])
    spike_probs = gmm.predict_proba(spike_train[:,2:])

    vanilla_gmm = np1_data_loader.prepare_decoder_input(
        np.c_[spike_times, spike_labels, spike_probs],
        is_gmm = True, n_t_bins = T, regional = True)
    
    print(f'When C = {c}:')
    y_train, y_test, _, _, acc = static_decoder(vanilla_gmm, advi_data_loader.behavior, kf_train_ids[0], kf_test_ids[0])
    val_accs.append(acc)

C = candidates[np.argmax(val_accs)]
gmm = GaussianMixture(n_components = C, 
                          covariance_type = 'full', 
                          init_params = 'k-means++')
gmm.fit(np.concatenate(trials)[:,2:])

print(f'Best C = {C}.')


ks_all_accs = []
thresh_accs = []
enc_accs = []
for i in range(1, len(kf_train_ids)):
    
    print(f'Started decoding {i} / 5 folds:')
    res_metrics = {}
    
    train_trials, train_trial_ids, train_time_ids, \
    test_trials, test_trial_ids, test_time_ids = advi_data_loader.split_train_test(kf_train_ids[i], kf_test_ids[i])
    
    lambdas, p = advi_data_loader.compute_lambda(gmm)
    
    y_train = torch.zeros(train_trials.shape[0])
    for k_idx, k in enumerate(advi_data_loader.train_ids):
        y_train[train_trial_ids == k] = advi_data_loader.train_behavior[k_idx]
    y_train = y_train.reshape(-1,1)
    
    print(f'Started fitting CAVI:')
    
    cavi = CAVI(
        init_mu = torch.tensor(gmm.means_), 
        init_cov = torch.tensor(gmm.covariances_), 
        init_lam = torch.tensor(lambdas), 
        train_k_ids = [torch.argwhere(torch.tensor(train_trial_ids) == k).reshape(-1) \
                       for k in advi_data_loader.train_ids], 
        train_t_ids = [torch.argwhere(torch.tensor(train_time_ids) == t).reshape(-1) for t in range(T)],
        test_k_ids = [torch.argwhere(torch.tensor(test_trial_ids) == k).reshape(-1) for k in advi_data_loader.test_ids],
        test_t_ids = [torch.argwhere(torch.tensor(test_time_ids) == t).reshape(-1) for t in range(T)]
    )
    
    enc_r, enc_lam, enc_mu, enc_cov, enc_elbo = cavi.encode(
        s = torch.tensor(train_trials[:,1:]),
        y = y_train, max_iter=30)
    
    dec_r, dec_nu, dec_mu, dec_cov, dec_p, dec_elbo = cavi.decode(
        s = torch.tensor(test_trials[:,1:]),
        init_p = torch.tensor([p]), 
        init_mu = enc_mu, 
        init_cov = enc_cov,
        init_lam = enc_lam, 
        test_k_ids = test_trial_ids, 
        test_ids = advi_data_loader.test_ids, 
        max_iter=30)
    
    print(f'Finished fitting CAVI.')
    
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
        is_gmm = True, n_t_bins = T, regional = True)
    
    y_train, y_test, _, _, acc = static_decoder(vanilla_gmm, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'gmm': [acc]})
    
    print(f'Started decoding using multi-unit thresholding:')
    
    thresholded = np1_data_loader.prepare_decoder_input(
        np.c_[spike_times, spike_channels],
        is_gmm = False, n_t_bins = T, regional = True)
    
    y_train, y_test, _, _, acc = static_decoder(thresholded, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'thresholded': [acc]})
    thresh_accs.append(acc)
    
    print(f'Started decoding using encoded GMM:')
    
    _, _ = cavi.eval_performance(dec_nu, advi_data_loader.test_behavior)
    
    y_pred = (1. * ( dec_nu > .5 )).numpy().astype(int)
    _, enc_all = cavi.encode_gmm(
        # advi_data_loader.trials, enc_lam.numpy(), enc_mu.numpy(), enc_cov.numpy(), 
        advi_data_loader.trials, enc_lam.numpy(), dec_mu.numpy(), dec_cov.numpy(), 
        train, test, y_train, y_pred)

    _, _, _, _, acc = static_decoder(enc_all, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'enc_gmm': [acc]})
    enc_accs.append(acc)
    
    print(f'Started decoding using all KS units:')
    
    ks_all = np1_data_loader.load_all_units(region=region)
    ks_all = np.concatenate(ks_all)

    all_units = np1_data_loader.prepare_decoder_input(
        ks_all, is_gmm = False, n_t_bins = T, regional = True)
    
    _, _, _, _, acc = static_decoder(all_units, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'all_units': [acc]})
    ks_all_accs.append(acc)
    
    print(f'Started decoding using good KS units:')
    
    ks_good = np1_data_loader.load_good_units(region=region)
    ks_good = np.concatenate(ks_good)

    good_units = np1_data_loader.prepare_decoder_input(
        ks_good, is_gmm = False, n_t_bins = T, regional = True)
    
    _, _, _, _, acc = static_decoder(good_units, advi_data_loader.behavior, train, test)
    
    res_metrics.update({'good_units': [acc]})
    
    np.save(f'../neurips_results/{session}/{region}/{behavior}/res_fold{i+1}.npy', res_metrics)
    
print(f'thresholded mean acc: {np.mean(thresh_accs):.3f}')
print(f'sorted mean acc: {np.mean(ks_all_accs):.3f}')
print(f'encoded MoG mean acc: {np.mean(enc_accs):.3f}')

