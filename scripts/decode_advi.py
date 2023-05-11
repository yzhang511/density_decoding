"""


"""
import os
import sys
import argparse
import numpy as np
import random

import torch
from sklearn.model_selection import KFold

from clusterless.utils import IBLDataLoader, ADVIDataLoader, initialize_gmm
from clusterless.advi import ADVI, train_advi, encode_gmm
from clusterless.decoder import (
        discrete_decoder, 
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
    g.add_argument("--pid")
    g.add_argument("--ephys_path")
    g.add_argument("--behavior_path", default=None)
    g.add_argument("--out_path")
    g.add_argument("--kilosort_feature_path")
    
    g = ap.add_argument_group("Decoding configuration")
    g.add_argument("--behavior", 
                   default="choice", 
                   type=str, 
                   choices=[
                       "choice", "motion_energy", "wheel_speed", "wheel_velocity",
                       "paw_speed", "pupil_diameter"
                   ])
    g.add_argument("--brain_region", default="all", type=str)
    g.add_argument("--n_time_bins", default=30, type=int)
    g.add_argument("--relocalize_kilosort", action="store_true")
    g.add_argument("--penalty_type", default="l2", type=str)
    g.add_argument("--penalty_strength", default=1000, type=int)
    g.add_argument("--sliding_window_size", default=7, type=int)
    g.add_argument("--train_with_motion_energy", action="store_true")
    g.add_argument("--featurize_behavior", action="store_true")
    
    g = ap.add_argument_group("Training configuration")
    g.add_argument("--batch_size", default=1, type=int)
    g.add_argument("--learning_rate", default=1e-3, type=float)
    g.add_argument("--max_iter", default=30, type=int)
    
    args = ap.parse_args()
    
    # -- load data 
    ibl_data_loader = IBLDataLoader(
        probe_id = args.pid, 
        ephys_path = args.ephys_path, 
        behavior_path = args.behavior_path
    )
    
    print(f"Decode {args.behavior} from the brain region {args.brain_region}:")
    
    if args.brain_region != "all":
        is_regional = True
    else:
        is_regional = False

    if args.relocalize_kilosort:
        pipeline_type = "relocalize_kilosort"
        if args.kilosort_feature_path != None:
            trials = ibl_data_loader.relocalize_kilosort(args.kilosort_feature_path, region=args.brain_region)
        else:
            print("Need path to the relocalized kilosort spike features.")
            sys.exit()
    else:
        pipeline_type = "our_pipeline"
        trials = ibl_data_loader.load_spike_features(region=args.brain_region)
        
    n_spikes = np.concatenate(trials).shape[0]
    print(f"# spikes available for decoding is {n_spikes}.")
        
    if args.behavior != "stimulus":
        behavior = ibl_data_loader.load_behaviors(args.behavior, featurize=args.featurize_behavior)
    else:
        # TO DO: include stimulus later.  
        print("Stimulus decoding is under development.")
        
    # -- prepare data for model training
    if args.train_with_motion_energy:
        encoding_behavior = ibl_data_loader.load_behaviors("motion_energy")
    else:
        encoding_behavior = behavior.copy()
        
    advi_data_loader = ADVIDataLoader(
                             data = trials, 
                             behavior = behavior, 
                             n_t_bins = args.n_time_bins
                       )
    gmm = initialize_gmm(np.concatenate(trials)[:,1:])
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
                y = torch.tensor(encoding_behavior), 
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
        
        spike_train = np.concatenate(trials)
        spike_times, spike_channels = spike_train[:,0], spike_train[:,1]
        # (spike_labels, spike_probs) needed for vanilla gmm
        # spike_labels = gmm.predict(spike_train[:,2:])
        # spike_probs = gmm.predict_proba(spike_train[:,2:])
        
        thresholded = ibl_data_loader.prepare_decoder_input(
            np.c_[spike_times, spike_channels],
            is_gmm=False, n_t_bins=n_t, regional=is_regional
        )
        saved_decoder_inputs.update({"thresholded": thresholded})
        fig_path = args.out_path + \
            f"/{args.pid}/{args.behavior}/{args.brain_region}/{pipeline_type}/plots/"
        os.makedirs(fig_path, exist_ok=True) 
        plot_decoder_input(thresholded, 'thresholded', len(spike_train), save_fig=True,
                           out_path=fig_path+f"thresholded_input_fold{i+1}.png")

        if args.behavior == "choice":
            y_train, y_test, y_pred, _, acc = discrete_decoder(
                thresholded, 
                advi_data_loader.behavior, 
                train, 
                test, 
                args.penalty_type,
                args.penalty_strength,
            )
            saved_metrics.update({"thresholded": acc})
            saved_y_obs.update({"thresholded": y_test})
            saved_y_pred.update({"thresholded": y_pred})
        else:
            if args.train_with_motion_energy:
                enc_behave_train, _, enc_behave_pred = continuous_decoder(
                    thresholded, encoding_behavior, train, test, args.penalty_strength
                )
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
        
        if args.behavior == "choice":
            _, y_test, y_pred, _, acc = discrete_decoder(
                encoded_weights, 
                advi_data_loader.behavior, 
                train, 
                test,
                args.penalty_type,
                args.penalty_strength
            )
            saved_metrics.update({"advi_gmm": acc})
            saved_y_obs.update({"advi_gmm": y_test})
            saved_y_pred.update({"advi_gmm": y_pred})
        else:
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

            
        if args.train_with_motion_energy:
            print("Decode using ADVI + GMM (train with motion energy):")
        
            encoded_pis, encoded_weights = encode_gmm(
                advi, advi_data_loader.trials, train, test, enc_behave_train, enc_behave_pred
            )
            saved_decoder_inputs.update({"advi_gmm_me": encoded_weights})
            saved_mixing_props.update({"advi_gmm_me": encoded_pis})
            plot_decoder_input(encoded_weights, 'ADVI + GMM', len(spike_train), save_fig=True,
                               out_path=fig_path+f"advi_gmm_me_input_fold{i+1}.png")


            if args.behavior == "choice":
                print("Cannot train ADVI with motion energy and then decode choice.")
            else:
                window_y_train, window_y_test, window_y_pred, r2, mse, corr = \
                    sliding_window_decoder(
                        encoded_weights, 
                        advi_data_loader.behavior, 
                        train, 
                        test,
                        window_size=args.sliding_window_size,
                        penalty_strength=args.penalty_strength
                    )
                saved_metrics.update({"advi_gmm_me": [r2, mse, corr]})
                saved_y_obs.update({"advi_gmm_me": window_y_test})
                saved_y_pred.update({"advi_gmm_me": window_y_pred})
                plot_behavior_traces(window_y_test, window_y_pred, prefix + " " + suffix, "ADVI + GMM",
                                     save_fig=True, out_path=fig_path+f"advi_gmm_me_traces_fold{i+1}.png")

            
        print(f'Decode using all Kilosort units:')

        all_units = np.concatenate(
            ibl_data_loader.load_all_units(region=args.brain_region)
        )
        ks_all = ibl_data_loader.prepare_decoder_input(
            all_units, is_gmm=False, n_t_bins=n_t, regional=is_regional
        )
        saved_decoder_inputs.update({"ks_all": ks_all})
        plot_decoder_input(ks_all, 'all ks units', len(all_units), save_fig=True,
                           out_path=fig_path+f"ks_all_input_fold{i+1}.png")
        
        if args.behavior == "choice":
            _, y_test, y_pred, _, acc = discrete_decoder(
                ks_all, 
                advi_data_loader.behavior, 
                train, 
                test,
                args.penalty_type,
                args.penalty_strength
            )
            saved_metrics.update({"ks_all": acc})
            saved_y_obs.update({"ks_all": y_test})
            saved_y_pred.update({"ks_all": y_pred})
        else:
            window_y_train, window_y_test, window_y_pred, r2, mse, corr = \
                sliding_window_decoder(
                    ks_all, 
                    advi_data_loader.behavior, 
                    train, 
                    test,
                    window_size=args.sliding_window_size,
                    penalty_strength=args.penalty_strength
                )
            saved_metrics.update({"ks_all": [r2, mse, corr]})
            saved_y_obs.update({"ks_all": window_y_test})
            saved_y_pred.update({"ks_all": window_y_pred})
            plot_behavior_traces(window_y_test, window_y_pred, prefix + " " + suffix, "KS all units",
                                 save_fig=True, out_path=fig_path+f"ks_all_traces_fold{i+1}.png")


        print(f'Decode using good Kilosort units:')

        try:
            good_units = np.concatenate(
                ibl_data_loader.load_good_units(region=args.brain_region)
            )
        except ValueError:
            print("Cannot decode since no good units found.")
            continue
            
        ks_good = ibl_data_loader.prepare_decoder_input(
            good_units, is_gmm=False, n_t_bins=n_t, regional=is_regional
        )
        saved_decoder_inputs.update({"ks_good": ks_good})
        plot_decoder_input(ks_good, 'good ks units', len(good_units), save_fig=True,
                           out_path=fig_path+f"ks_good_input_fold{i+1}.png")
        
        if args.behavior == "choice":
            _, y_test, y_pred, _, acc = discrete_decoder(
                ks_good, 
                advi_data_loader.behavior, 
                train, 
                test,
                args.penalty_type,
                args.penalty_strength
            )
            saved_metrics.update({"ks_good": acc})
            saved_y_obs.update({"ks_good": y_test})
            saved_y_pred.update({"ks_good": y_pred})
        else:
            window_y_train, window_y_test, window_y_pred, r2, mse, corr = \
                sliding_window_decoder(
                    ks_good, 
                    advi_data_loader.behavior, 
                    train, 
                    test,
                    window_size=args.sliding_window_size,
                    penalty_strength=args.penalty_strength
                )
            saved_metrics.update({"ks_good": [r2, mse, corr]})
            saved_y_obs.update({"ks_good": window_y_test})
            saved_y_pred.update({"ks_good": window_y_pred})
            prefix, suffix = args.behavior.split("_")
            plot_behavior_traces(window_y_test, window_y_pred, prefix + " " + suffix, "KS good units",
                                 save_fig=True, out_path=fig_path+f"ks_good_traces_fold{i+1}.png")
           
        # -- save outputs
        save_path = {}
        for res in ["metrics", "y_obs", "y_pred", "decoder_inputs", "mixing_props"]:
            save_path.update({res: args.out_path + 
                              f"/{args.pid}/{args.behavior}/{args.brain_region}/{pipeline_type}/{res}/"})
            os.makedirs(save_path[res], exist_ok=True) 
            
        np.save(save_path["metrics"] + f"fold{i+1}.npy", saved_metrics)
        np.save(save_path["y_obs"] + f"fold{i+1}.npy", saved_y_obs)
        np.save(save_path["y_pred"] + f"fold{i+1}.npy", saved_y_pred)
        np.save(save_path["decoder_inputs"] + f"fold{i+1}.npy", saved_decoder_inputs)
        np.save(save_path["mixing_props"] + f"fold{i+1}.npy", saved_mixing_props)

    
    