"""


"""
import os
import sys
import argparse
import numpy as np
import random

import torch
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from clusterless.utils import NP1DataLoader, ADVIDataLoader, initialize_gmm
from clusterless.cavi import CAVI
from clusterless.decoder import discrete_decoder 

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
    g.add_argument("--behavior_path")
    g.add_argument("--geom_path")
    g.add_argument("--out_path")
    g.add_argument("--kilosort_feature_path")
    
    g = ap.add_argument_group("Decoding configuration")
    g.add_argument("--brain_region", 
                   default="all", 
                   type=str, 
                   choices=[
                       "all", "po", "lp", "dg", "ca1", "vis"
                   ])
    g.add_argument("--n_time_bins", default=10, type=int)
    g.add_argument("--relocalize_kilosort", action="store_true")
    
    g = ap.add_argument_group("Training configuration")
    g.add_argument("--max_iter", default=3, type=int)
    
    args = ap.parse_args()
    
    # -- load data 
    np1_data_loader = NP1DataLoader(
        probe_id = args.pid, 
        geom_path = args.geom_path, 
        ephys_path = args.ephys_path, 
        behavior_path = args.behavior_path
    )
    
    print(f"Decode binary choice from the brain region {args.brain_region}:")
    
    if args.brain_region != "all":
        is_regional = True

    if args.relocalize_kilosort:
        if args.kilosort_feature_path != None:
            trials = np1_data_loader.relocalize_kilosort(args.kilosort_feature_path, region=args.brain_region)
        else:
            print("Need path to the relocalized kilosort spike features.")
            sys.exit()
    else:
        trials = np1_data_loader.load_spike_features(region=args.brain_region)

    behavior = np1_data_loader.load_behaviors("choice")
        
    # -- prepare data for model training
    cavi_data_loader = ADVIDataLoader(
                             data = trials, 
                             behavior = behavior, 
                             n_t_bins = args.n_time_bins
                       )
    gmm = initialize_gmm(np.concatenate(trials)[:,1:])
    n_t = cavi_data_loader.n_t_bins
    n_c = gmm.means_.shape[0]
    n_d = gmm.means_.shape[1]
    
    gmm = GaussianMixture(n_components = n_c, 
                          covariance_type = 'full', 
                          init_params = 'k-means++')
    gmm.fit(np.concatenate(trials)[:,2:])
    print(f"Initializ GMM with {n_c} components and {n_d} spike features.")
    
    
    # -- k-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (train, test) in enumerate(kf.split(cavi_data_loader.behavior)):

        print(f"Fold {i+1} / 5:")
        saved_metrics, saved_y_obs, saved_y_pred = {}, {}, {}
        saved_decoder_inputs, saved_mixing_props = {}, {}

        train_data, train_ks, train_ts, test_data, test_ks, test_ts = \
            cavi_data_loader.split_train_test(train, test)
        
        init_lam, init_p = cavi_data_loader.compute_lambda(gmm)
        
        y_train = torch.zeros(train_data.shape[0]).reshape(-1,1)
        for i, k in enumerate(train):
            y_train[train_ks == k] = cavi_data_loader.train_behavior[i]

        cavi = CAVI(
                init_mu = gmm.means_, 
                init_cov = gmm.covariances_, 
                init_lam = init_lam, 
                train_ks = [torch.argwhere(torch.tensor(train_ks)==k).reshape(-1) for k in train], 
                train_ts = [torch.argwhere(torch.tensor(train_ts)==t).reshape(-1) for t in range(n_t)],
                test_ks = [torch.argwhere(torch.tensor(test_ks)==k).reshape(-1) for k in test],
                test_ts = [torch.argwhere(torch.tensor(test_ts) == t).reshape(-1) for t in range(n_t)]
        )
        
        print(f"Encode:")
        encoded_r, encoded_lam, encoded_mu, encoded_cov, elbo = cavi.encode(
            s = train_data[:,1:],
            y = y_train, 
            max_iter = args.max_iter
        )
    
        print(f"Decode:")
        decoded_r, decoded_nu, decoded_mu, decoded_cov, decoded_p, elbo = cavi.decode(
            s = test_data[:,1:],
            init_p = init_p, 
            init_mu = encoded_mu, 
            init_cov = encoded_cov,
            init_lam = encoded_lam, 
            test_ks = test_ks, 
            test_ids = test, 
            max_iter = args.max_iter
        )
        _, _ = cavi.eval_perf(decoded_nu, cavi_data_loader.behavior[test])
        
        print("Decode using multi-unit thresholding:")
        
        spike_train = np.concatenate(trials)
        spike_times, spike_channels = spike_train[:,0], spike_train[:,1]
        spike_labels = gmm.predict(spike_train[:,2:])
        spike_probs = gmm.predict_proba(spike_train[:,2:])
        
        thresholded = np1_data_loader.prepare_decoder_input(
            np.c_[spike_times, spike_channels],
            is_gmm=False, n_t_bins=n_t, regional=is_regional
        )
        saved_decoder_inputs.update({"thresholded": thresholded})

        y_train, y_test, y_pred, _, acc = discrete_decoder(
            thresholded, cavi_data_loader.behavior, train, test
        )
        saved_metrics.update({"thresholded": acc})
        saved_y_obs.update({"thresholded": y_test})
        saved_y_pred.update({"thresholded": y_pred})

            
        print("Decode using CAVI + GMM:")
        
        try:
            encoded_pis, encoded_weights = cavi.encode_gmm(
                cavi_data_loader.trials,
                encoded_lam.numpy(), 
                decoded_mu.numpy(), 
                decoded_cov.numpy(), 
                train, test, y_train, y_pred
            )
        except np.linalg.LinAlgError:
            encoded_pis, encoded_weights = cavi.encode_gmm(
                cavi_data_loader.trials,
                encoded_lam.numpy(), 
                gmm.means_, 
                gmm.covariances_, 
                train, test, y_train, y_pred
            )
            
        saved_decoder_inputs.update({"cavi_gmm": encoded_weights})
        saved_mixing_props.update({"cavi_gmm": encoded_pis})
        
        _, y_test, y_pred, _, acc = discrete_decoder(
            encoded_weights, cavi_data_loader.behavior, train, test
        )
        saved_metrics.update({"cavi_gmm": acc})
        saved_y_obs.update({"cavi_gmm": y_test})
        saved_y_pred.update({"cavi_gmm": y_pred})

            
        print(f'Decode using all Kilosort units:')

        all_units = np.concatenate(
            np1_data_loader.load_all_units(region=args.brain_region)
        )
        ks_all = np1_data_loader.prepare_decoder_input(
            all_units, is_gmm=False, n_t_bins=n_t, regional=is_regional
        )
        saved_decoder_inputs.update({"ks_all": ks_all})
        
        _, y_test, y_pred, _, acc = discrete_decoder(
            ks_all, cavi_data_loader.behavior, train, test
        )
        saved_metrics.update({"ks_all": acc})
        saved_y_obs.update({"ks_all": y_test})
        saved_y_pred.update({"ks_all": y_pred})


        print(f'Decode using good Kilosort units:')

        good_units = np.concatenate(
            np1_data_loader.load_good_units(region=args.brain_region)
        )
        ks_good = np1_data_loader.prepare_decoder_input(
            good_units, is_gmm=False, n_t_bins=n_t, regional=is_regional
        )
        saved_decoder_inputs.update({"ks_good": ks_good})
        
        _, y_test, y_pred, _, acc = discrete_decoder(
            ks_good, cavi_data_loader.behavior, train, test
        )
        saved_metrics.update({"ks_good": acc})
        saved_y_obs.update({"ks_good": y_test})
        saved_y_pred.update({"ks_good": y_pred})
           
        # -- save outputs
        save_path = {}
        for res in ["metrics", "y_obs", "y_pred", "decoder_inputs", "mixing_props"]:
            if args.relocalize_kilosort:
                save_path.update({res: args.out_path + 
                    f"/{args.pid}/choice/{args.brain_region}/relocalize_kilosort/cavi_{res}/"})
            else:
                save_path.update({res: args.out_path + 
                    f"/{args.pid}/choice/{args.brain_region}/our_pipeline/cavi_{res}/"})
            os.makedirs(save_path[res], exist_ok=True) 
            
        np.save(save_path["metrics"] + f"fold{i+1}.npy", saved_metrics)
        np.save(save_path["y_obs"] + f"fold{i+1}.npy", saved_y_obs)
        np.save(save_path["y_pred"] + f"fold{i+1}.npy", saved_y_pred)
        np.save(save_path["decoder_inputs"] + f"fold{i+1}.npy", saved_decoder_inputs)
        np.save(save_path["mixing_props"] + f"fold{i+1}.npy", saved_mixing_props)
        
        
