"""


"""
import sys
import argparse
import numpy as np
import random
import torch
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from clusterless.utils import NP1DataLoader, ADVIDataLoader, initialize_gmm
from clusterless.advi import advi
from clusterless.cavi import cavi
from clusterless.decoder import (
    discrete_decoder, 
    continuous_decoder, 
    sliding_window, 
    sliding_window_decoder
    )

def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.set_default_dtype(torch.double)



if __name__ == "__main__":
    
    set_seed(666)
    
    # -- args
    parser = argparse.ArgumentParser()
    
    g = parser.add_argument_group("Data input/output")
    g.add_argument("--pid")
    g.add_argument("--ephys_path")
    g.add_argument("--behavior_path")
    g.add_argument("--geom_path")
    g.add_argument("--output_path")
    
    g = parser.add_argument_group("Decoding configuration")
    g.add_argument("--behavior", 
                   default="choice", 
                   type=str, 
                   choices=[
                       "choice", "motion_energy", "wheel_speed", "wheel_velocity"
                   ])
    g.add_argument("--brain_region", 
                   default="all", 
                   type=str, 
                   choices=[
                       "all", "po", "lp", "dg", "ca1", "vis"
                   ])
    g.add_argument("--n_time_bins", default=30, type=int)
    g.add_argument("--unsort_kilosort", action="store_true")
    g.add_argument("--relocalize_kilosort", action="store_true")
    g.add_argument("--kilosort_feature_path")
    g.add_argument("--cavi", action="store_true")
    
    args = parser.parse_args()
    
    if args.cavi & args.behavior != "choice":
        print("CAVI can only be used for decoding binary choice.")
        sys.exit()
    
    # -- load data 
    np1_data_loader = NP1DataLoader(
        probe_id = args.pid, 
        geom_path = args.geom_path, 
        ephys_path = args.ephys_path, 
        behavior_path = args.behavior_path
    )
    
    print(f"Decode {args.behavior} from the brain region {args.brain_region}:")

    if args.relocalize_kilosort:
        if args.kilosort_feature_path != None:
            trials = np1_data_loader.relocalize_kilosort(args.kilosort_feature_path, region=args.brain_region)
        else:
            print("Need to enter path to the relocalized kilosort spike features.")
            sys.exit()
    else:
        trials = np1_data_loader.load_spike_features(region=args.brain_region)

    if args.behavior != "stimulus":
        behavior = np1_data_loader.load_behaviors(args.behavior)
    else:
        # TO DO: include stimulus later.  
        print('Stimulus is currently not decodable.')

    # -- prepare data for model training
    advi_data_loader = ADVIDataLoader(
                             data = trials, 
                             behavior = behavior, 
                             n_t_bins = args.n_time_bins
                       )
    gmm = initialize_gmm(np.concatenate(trials)[:,1:])
    Nt = advi_data_loader.n_t_bins
    Nc = gmm.means_.shape[0]
    Nd = gmm.means_.shape[1]
    
    
    
    