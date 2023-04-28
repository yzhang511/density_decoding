import os
import sys
import argparse
import numpy as np
import random

import torch
from scipy.io import loadmat
from sklearn.model_selection import KFold

from clusterless.utils import NP1NHPDataLoader, ADVIDataLoader, initialize_gmm
from clusterless.advi import ADVI, train_advi, encode_gmm
from clusterless.decoder import (
        continuous_decoder, 
        sliding_window, 
        sliding_window_decoder
    )
from clusterless.viz import plot_decoder_input, plot_behavior_traces


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_default_dtype(torch.double)



if __name__ == "__main__":
    
    seed = 666
    set_seed(seed)
    
    # -- args
    ap = argparse.ArgumentParser()
    
    g = ap.add_argument_group("Data input/output")
    g.add_argument("--ephys_path")
    g.add_argument("--behavior_path")
    g.add_argument("--trial_times_path")
    g.add_argument("--out_path")
    g.add_argument("--sampling_freq", default=30_000)
    
    g = ap.add_argument_group("Decoding configuration")
    g.add_argument("--behavior", 
                   default="y", 
                   type=str, 
                   choices=[
                       "x", "y", "z"
                   ])
    g.add_argument("--n_time_bins", default=30, type=int)
    g.add_argument("--t_start", default=0., type=float)
    g.add_argument("--t_end", default=9.85, type=float)
    g.add_argument("--trials", default=50, type=int)
    g.add_argument("--penalty_type", default="l2", type=str)
    g.add_argument("--penalty_strength", default=1000, type=int)
    g.add_argument("--sliding_window_size", default=7, type=int)
    
    g = ap.add_argument_group("Training configuration")
    g.add_argument("--batch_size", default=1, type=int)
    g.add_argument("--learning_rate", default=1e-3, type=float)
    g.add_argument("--max_iter", default=20, type=int)
    
    args = ap.parse_args()
    
    # -- load data 
    spike_index = np.load(args.ephys_path + "spike_index.npy")
    localization_results = np.load(args.ephys_path + "localization_results.npy")
    ephys_data = np.c_[spike_index, localization_results]
    ephys_data[:,0] = ephys_data[:,0] / args.sampling_freq
    trial_times = loadmat(args.trial_times_path)["simTime"]
    behavior = loadmat(args.behavior_path)["force"]

    np1_nhp_loader = NP1NHPDataLoader(ephys_data, behavior, trial_times, 
                                      n_t_bins=args.n_time_bins, t_start=args.t_start, t_end=args.t_end)
    
    print(f"Decode {args.behavior}-axis reaching force from {args.trials} trials.")
    
    bin_behavior, active_trials = np1_nhp_loader.bin_behaviors()
    active_trials = active_trials[:args.trials]
    bin_behavior = bin_behavior[active_trials]
    trials = np1_nhp_loader.partition_into_trials(active_trials)
    n_spikes = np.concatenate(trials).shape[0]
    print(f"# spikes available for decoding is {n_spikes}.")
    
    if args.behavior == "x":
        behavior = bin_behavior[:,:,0]
    elif args.behavior == "y":
        behavior = bin_behavior[:,:,1]
    elif args.behavior == "z":
        behavior = bin_behavior[:,:,-1]
        
    advi_data_loader = ADVIDataLoader(
                         data = trials, 
                         behavior = behavior, 
                         n_t_bins = args.n_time_bins,
                         t_start = args.t_start, t_end = args.t_end
                   )
    # NP1-NHP data contains too many neurons so choose to not split
    gmm = initialize_gmm(np.concatenate(trials)[:,1:], verbose=False, split=False)
    n_t = advi_data_loader.n_t_bins
    n_c = gmm.means_.shape[0]
    n_d = gmm.means_.shape[1]
    print(f"Initializ GMM with {n_c} components and {n_d} spike features.")
    
    # -- k-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (train, test) in enumerate(kf.split(advi_data_loader.behavior)):

        print(f"Fold {i+1} / 5:")
        saved_metrics, saved_y_obs, saved_y_pred = {}, {}, {}
        saved_decoder_inputs, saved_mixing_props = {}, {}

        train_data, train_ks, train_ts, test_data, test_ks, test_ts = \
            advi_data_loader.split_train_test(train, test)

        print(f"Train ADVI with a batch size of {args.batch_size}:")

        advi = ADVI(
            n_k=args.batch_size, 
            n_t=n_t, 
            n_c=n_c, 
            n_d=n_d, 
            init_means=gmm.means_, 
            init_covs=gmm.covariances_
        )
        
        try:
            elbos = train_advi(
                advi,
                s = torch.tensor(train_data[:,1:]), 
                y = torch.tensor(advi_data_loader.behavior), 
                ks = torch.tensor(train_ks), 
                ts = torch.tensor(train_ts), 
                batch_ids = list(zip(*(iter(train),) * args.batch_size)), 
                optim = torch.optim.Adam(advi.parameters(), lr=args.learning_rate), 
                max_iter=args.max_iter
            )
        except ValueError:
            print("Encountered numerical errors during gradient update. Go to the next fold.")
            continue
        
        
        print("Decode using multi-unit thresholding:")
        
        thresholded = np1_nhp_loader.prepare_decoder_input(trials, active_trials)
        saved_decoder_inputs.update({"thresholded": thresholded})
        fig_path = args.out_path + f"/np1_nhp/{args.behavior}/plots/"
        os.makedirs(fig_path, exist_ok=True) 
        plot_decoder_input(thresholded, 'thresholded', len(spike_train), save_fig=True,
                           out_path=fig_path+f"thresholded_input_fold{i+1}.png")

        y_train, _, y_pred = continuous_decoder(
            thresholded, advi_data_loader.behavior, train, test, args.penalty_strength
        )
        window_y_train, window_y_test, window_y_pred, r2, mse, corr = \
            sliding_window_decoder(
                thresholded, 
                advi_data_loader.behavior, 
                train, 
                test,
                window_size=args.sliding_window_size,
                penalty_strength=args.penalty_strength
            )
        saved_metrics.update({"thresholded": [r2, mse, corr]})
        saved_y_obs.update({"thresholded": window_y_test})
        saved_y_pred.update({"thresholded": window_y_pred})
        prefix, suffix = args.behavior.split("_")
        plot_behavior_traces(window_y_test, window_y_pred, prefix + " " + suffix, "thresholded",
                             save_fig=True, out_path=fig_path+f"thresholded_traces_fold{i+1}.png")
            
            
        print("Decode using ADVI + GMM:")
        
        encoded_pis, encoded_weights = encode_gmm(
            advi, advi_data_loader.trials, train, test, y_train, y_pred
        )
        saved_decoder_inputs.update({"advi_gmm": encoded_weights})
        saved_mixing_props.update({"advi_gmm": encoded_pis})
        plot_decoder_input(encoded_weights, 'ADVI + GMM', len(spike_train), save_fig=True,
                           out_path=fig_path+f"advi_gmm_input_fold{i+1}.png")
        
        window_y_train, window_y_test, window_y_pred, r2, mse, corr = \
            sliding_window_decoder(
                encoded_weights, 
                advi_data_loader.behavior, 
                train, 
                test,
                window_size=args.sliding_window_size,
                penalty_strength=args.penalty_strength
            )
        saved_metrics.update({"advi_gmm": [r2, mse, corr]})
        saved_y_obs.update({"advi_gmm": window_y_test})
        saved_y_pred.update({"advi_gmm": window_y_pred})
        plot_behavior_traces(window_y_test, window_y_pred, prefix + " " + suffix, "ADVI + GMM",
                             save_fig=True, out_path=fig_path+f"advi_gmm_traces_fold{i+1}.png")

        # -- save outputs
        save_path = {}
        for res in ["metrics", "y_obs", "y_pred", "decoder_inputs", "mixing_props"]:
            save_path.update({res: args.out_path + 
                              f"/np1_nhp/{args.behavior}/{res}/"})
            os.makedirs(save_path[res], exist_ok=True) 
            
        np.save(save_path["metrics"] + f"fold{i+1}.npy", saved_metrics)
        np.save(save_path["y_obs"] + f"fold{i+1}.npy", saved_y_obs)
        np.save(save_path["y_pred"] + f"fold{i+1}.npy", saved_y_pred)
        np.save(save_path["decoder_inputs"] + f"fold{i+1}.npy", saved_decoder_inputs)
        np.save(save_path["mixing_props"] + f"fold{i+1}.npy", saved_mixing_props)

    
    