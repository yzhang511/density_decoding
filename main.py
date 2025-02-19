"""run decoding pipeline via command line."""

import os
import argparse
import random
import numpy as np
import torch
from pathlib import Path

from sklearn.model_selection import KFold

from density_decoding.utils.utils import set_seed
from density_decoding.utils.data_utils import IBLDataLoader
from density_decoding.decoders.behavior_decoder import (
    generic_decoder,
    sliding_window_decoder
)
from density_decoding.decode_pipeline import decode_pipeline


if __name__ == "__main__":
    
    seed = 666
    set_seed(seed)
    
    # -- args
    ap = argparse.ArgumentParser()
    
    g = ap.add_argument_group("Data Input/Output")
    g.add_argument("--pid")
    g.add_argument("--ephys_path")
    g.add_argument("--out_path")
    g.add_argument("--prior_path", default=None, type=str)
    
    g = ap.add_argument_group("Decoding Config")
    g.add_argument("--behavior", 
                   default="choice", 
                   type=str, 
                   choices=[
                       "choice", 
                       "prior",
                       "motion_energy", 
                       "wheel_speed", 
                       "wheel_velocity"
                   ])
    g.add_argument("--brain_region", default="all", type=str)
    g.add_argument("--n_t_bins", default=30, type=int)
    g.add_argument("--sliding_window_size", default=7, type=int)
    
    g = ap.add_argument_group("Model Training Config")
    g.add_argument("--batch_size", default=32, type=int)
    g.add_argument("--learning_rate", default=1e-3, type=float)
    g.add_argument("--max_iter", default=5000, type=int)
    g.add_argument("--fast_compute", action='store_false', default=True)
    g.add_argument("--stochastic", action='store_false', default=True)
    g.add_argument("--device", default="cpu", type=str, choices=["cpu", "gpu"])
    g.add_argument("--n_workers", default=1, type=int)
    
    args = ap.parse_args()
    
    
    # -- setup
    device = torch.device("cuda") if args.device == "gpu" else torch.device("cpu")
    
    behavior_type = "discrete" if args.behavior == "choice" else "continuous"
    
    
    # -- load data
    ibl_data_loader = IBLDataLoader(
        args.pid, 
        n_t_bins = args.n_t_bins,
        prior_path = args.prior_path
    )
    
    print("available brain regions to decode:")
    ibl_data_loader.check_available_brain_regions()
    
    behavior = ibl_data_loader.process_behaviors(args.behavior)
    
    ephys_path = Path(args.ephys_path)
    spike_index = np.load(ephys_path / "spike_index.npy")
    spike_features = np.load(ephys_path / "localization_results.npy")
    spike_times, spike_channels = spike_index.T
    
    bin_spike_features, bin_trial_idxs, bin_time_idxs = \
        ibl_data_loader.load_spike_features(
            spike_times, spike_channels, spike_features, args.brain_region
    )
    
    thresholded_spike_count = ibl_data_loader.load_thresholded_units(
        spike_times, spike_channels, args.brain_region
    )
    
    all_sorted_spike_count = ibl_data_loader.load_all_sorted_units(args.brain_region)
    
    try:
        good_sorted_spike_count = ibl_data_loader.load_good_sorted_units(args.brain_region)
        skip_good_ks = False
    except:
        skip_good_ks = True
        print("no good Kilosort units found in this brain region.")
    
    
    # -- CV
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (train, test) in enumerate(kf.split(behavior)):

        print(f"Fold {i+1} / 5:")
        
        saved_metrics, saved_y_obs, saved_y_pred, saved_y_prob = {}, {}, {}, {}

        weight_matrix = decode_pipeline(
            ibl_data_loader,
            bin_spike_features,
            bin_trial_idxs,
            bin_time_idxs,
            bin_behaviors = behavior,
            behavior_type = behavior_type,
            train = train,
            test = test,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            fast_compute=args.fast_compute,
            stochastic=args.stochastic,
            device=device,
            n_workers=args.n_workers
        )
        
        if np.logical_and(behavior_type == "continuous", args.behavior != "prior"):
            
            print("thresholded:")
            y_train, y_test, y_pred, metrics = sliding_window_decoder(
                thresholded_spike_count, behavior, train, test, behavior_type=behavior_type, verbose=True
            )
            saved_metrics.update({"thresholded": [metrics["r2"], metrics["mse"], metrics["corr"]]})
            saved_y_obs.update({"thresholded": y_test})
            saved_y_pred.update({"thresholded": y_pred})
            
            print("density-based:")
            y_train, y_test, y_pred, metrics = sliding_window_decoder(
                weight_matrix, behavior, train, test, behavior_type=behavior_type, verbose=True
            )
            saved_metrics.update({"density_based": [metrics["r2"], metrics["mse"], metrics["corr"]]})
            saved_y_obs.update({"density_based": y_test})
            saved_y_pred.update({"density_based": y_pred})
            
            print("all Kilosort units:")
            y_train, y_test, y_pred, metrics = sliding_window_decoder(
                all_sorted_spike_count, behavior, train, test, behavior_type=behavior_type, verbose=True
            )
            saved_metrics.update({"all_ks": [metrics["r2"], metrics["mse"], metrics["corr"]]})
            saved_y_obs.update({"all_ks": y_test})
            saved_y_pred.update({"all_ks": y_pred})
            
            if not skip_good_ks:
                print("good Kilosort units:")
                y_train, y_test, y_pred, metrics = sliding_window_decoder(
                    good_sorted_spike_count, behavior, train, test, behavior_type=behavior_type, verbose=True
                )
                saved_metrics.update({"good_ks": [metrics["r2"], metrics["mse"], metrics["corr"]]})
                saved_y_obs.update({"good_ks": y_test})
                saved_y_pred.update({"good_ks": y_pred})
        
            
        elif behavior_type == "discrete":
            
            print("thresholded:")
            y_train, y_test, y_pred, y_prob, metrics = generic_decoder(
                thresholded_spike_count, behavior, train, test, 
                behavior_type=behavior_type, verbose=True, return_prob=True
            )
            saved_metrics.update({"thresholded": metrics["acc"]})
            saved_y_obs.update({"thresholded": y_test})
            saved_y_pred.update({"thresholded": y_pred})
            saved_y_prob.update({"thresholded": y_prob})
            
            print("density-based:")
            y_train, y_test, y_pred, y_prob, metrics = generic_decoder(
                weight_matrix, behavior, train, test, 
                behavior_type=behavior_type, verbose=True, return_prob=True
            )
            saved_metrics.update({"density_based": metrics["acc"]})
            saved_y_obs.update({"density_based": y_test})
            saved_y_pred.update({"density_based": y_pred})
            saved_y_prob.update({"density_based": y_prob})
            
            print("all Kilosort units:")
            y_train, y_test, y_pred, y_prob, metrics = generic_decoder(
                all_sorted_spike_count, behavior, train, test, 
                behavior_type=behavior_type, verbose=True, return_prob=True
            )
            saved_metrics.update({"all_ks": metrics["acc"]})
            saved_y_obs.update({"all_ks": y_test})
            saved_y_pred.update({"all_ks": y_pred})
            saved_y_prob.update({"all_ks": y_prob})
            
            if not skip_good_ks:
                print("good Kilosort units:")
                y_train, y_test, y_pred, y_prob, metrics = generic_decoder(
                    good_sorted_spike_count, behavior, train, test, 
                    behavior_type=behavior_type, verbose=True, return_prob=True
                )
                saved_metrics.update({"good_ks": metrics["acc"]})
                saved_y_obs.update({"good_ks": y_test})
                saved_y_pred.update({"good_ks": y_pred})
                saved_y_prob.update({"good_ks": y_prob})
                
        elif args.behavior == "prior":
            
            print("thresholded:")
            y_train, y_test, y_pred, metrics = generic_decoder(
                thresholded_spike_count, behavior, train, test, behavior_type=behavior_type, verbose=True
            )
            saved_metrics.update({"thresholded": [metrics["r2"], metrics["mse"], metrics["corr"]]})
            saved_y_obs.update({"thresholded": y_test})
            saved_y_pred.update({"thresholded": y_pred})
            
            print("density-based:")
            y_train, y_test, y_pred, metrics = generic_decoder(
                weight_matrix, behavior, train, test, behavior_type=behavior_type, verbose=True
            )
            saved_metrics.update({"density_based": [metrics["r2"], metrics["mse"], metrics["corr"]]})
            saved_y_obs.update({"density_based": y_test})
            saved_y_pred.update({"density_based": y_pred})
            
            print("all Kilosort units:")
            y_train, y_test, y_pred, metrics = generic_decoder(
                all_sorted_spike_count, behavior, train, test, behavior_type=behavior_type, verbose=True
            )
            saved_metrics.update({"all_ks": [metrics["r2"], metrics["mse"], metrics["corr"]]})
            saved_y_obs.update({"all_ks": y_test})
            saved_y_pred.update({"all_ks": y_pred})
            
            if not skip_good_ks:
                print("good Kilosort units:")
                y_train, y_test, y_pred, metrics = generic_decoder(
                    good_sorted_spike_count, behavior, train, test, behavior_type=behavior_type, verbose=True
                )
                saved_metrics.update({"good_ks": [metrics["r2"], metrics["mse"], metrics["corr"]]})
                saved_y_obs.update({"good_ks": y_test})
                saved_y_pred.update({"good_ks": y_pred})
            
        # -- save outputs
        save_path = {}
        out_path = Path(args.out_path)
        for res in ["metrics", "y_obs", "y_pred", "y_prob", "trial_idx"]:
            save_path.update({res: out_path/args.pid/args.behavior/args.brain_region/res})
            os.makedirs(save_path[res], exist_ok=True) 
            
        np.save(save_path["metrics"] / f"fold_{i+1}.npy", saved_metrics)
        np.save(save_path["y_obs"] / f"fold_{i+1}.npy", saved_y_obs)
        np.save(save_path["y_pred"] / f"fold_{i+1}.npy", saved_y_pred)
        np.save(save_path["trial_idx"] / f"fold_{i+1}.npy", test)
        
        if behavior_type == "discrete":
            np.save(save_path["y_prob"] / f"fold_{i+1}.npy", saved_y_prob)
            
            
