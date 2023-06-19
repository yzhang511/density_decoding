import os
import random
import numpy as np
import torch
import warnings

from density_decoding.utils.utils import set_seed, to_device
from density_decoding.utils.data_utils import initilize_gaussian_mixtures

from density_decoding.models.advi import (
    ModelDataLoader, 
    ADVI, 
    train_advi,
    compute_posterior_weight_matrix, 
)

from density_decoding.models.cavi import (
    CAVI, 
    compute_lambda_for_cavi,
    compute_cavi_weight_matrix
)

from density_decoding.decoders.behavior_decoder import generic_decoder

seed = 666
set_seed(seed)


def decode_pipeline(
    data_loader, 
    bin_spike_features,
    bin_trial_idxs,
    bin_time_idxs,
    thresholded_spike_count,
    bin_behaviors,
    behavior_type,
    train,
    test,
    gmm_init_method="isosplit",
    inference="advi",
    batch_size=1,
    learning_rate=1e-2,
    max_iter=500,
    cavi_max_iter=10,
    fast_compute=True,
    stochastic=True,
    penalty_strength=1,
    device=torch.device("cpu"),
    n_workers=4
):
    """Run the decoding pipeline."""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        valid_types = ["discrete", "continuous"]
        assert behavior_type in valid_types, f"invalid behavior type; expected one of {valid_types}."
        
        valid_inf = ["advi", "cavi"]
        assert inference in valid_inf, f"invalid inference type; expected one of {valid_inf}."
        
        fast_compute = False if data_loader.type == "custom" else fast_compute
        
        y_train, _, y_pred, _ = generic_decoder(
            thresholded_spike_count, 
            bin_behaviors, 
            train, 
            test, 
            behavior_type=behavior_type,
            penalty_strength=penalty_strength,
            seed=seed
        )

        spike_features = np.concatenate(
            np.concatenate(bin_spike_features)
        )

        gmm = initilize_gaussian_mixtures(
            spike_features=spike_features[:,1:], 
            spike_channels=spike_features[:,0], 
            method=gmm_init_method, 
            verbose=False
        )
        n_t = data_loader.n_t_bins
        n_c = gmm.means_.shape[0]
        n_d = gmm.means_.shape[1]
        print(f"Initialized a mixture with {n_c} components.")

        model_data_loader = ModelDataLoader(
            bin_spike_features,
            bin_behaviors,
            bin_trial_idxs,
            bin_time_idxs
        )

        if behavior_type == "discrete":
            max_iter = 50
            model_data_loader.bin_behaviors = model_data_loader.bin_behaviors.reshape(-1,1)
            
        train_spike_features, train_trial_idxs, train_time_idxs, \
        test_spike_features, test_trial_idxs, test_time_idxs = \
        model_data_loader.split_train_test(train, test)

        if inference == "advi":
            
            advi = ADVI(
                n_t=n_t, 
                gmm=gmm, 
                device=device
            )
            
            if batch_size == 1:
                batch_idxs = list(zip(*(iter(train),) * batch_size))
            else:
                batch_idxs = list(zip(*(iter(np.arange(len(train))),) * batch_size))
            
            elbos = train_advi(
                advi,
                spike_features = to_device(train_spike_features[:,1:], device), 
                behaviors = to_device(model_data_loader.bin_behaviors, device), 
                trial_idxs = to_device(train_trial_idxs, device), 
                time_idxs = to_device(train_time_idxs, device), 
                batch_idxs= batch_idxs, 
                optim = torch.optim.Adam(advi.parameters(), lr=learning_rate),
                max_iter=max_iter,
                fast_compute=fast_compute,
                stochastic=stochastic
            )
            
            parameters = advi.parameters()
            detached_parameters = [param.detach_() for param in parameters]
            post_params = {
                "b": advi.b.loc.numpy(),
                "beta": advi.beta.loc.numpy(),
                "means": advi.means.numpy(),
                "covs": advi.covs.numpy(),
            }

            mixture_weights, weight_matrix = compute_posterior_weight_matrix(
                bin_spike_features, y_train, y_pred, train, test, post_params, n_workers
            )

        else:
            
            init_lam, init_p = compute_lambda_for_cavi(bin_spike_features, bin_behaviors, gmm)

            train_behaviors = torch.zeros(train_spike_features.shape[0]).reshape(-1,1)
            for i, k in enumerate(train):
                train_behaviors[train_trial_idxs==k] = torch.tensor(model_data_loader.train_y[i])

            cavi = CAVI(
                init_means = gmm.means_, 
                init_covs = gmm.covariances_, 
                init_lambdas = init_lam, 
                train_trial_idxs = [torch.argwhere(
                    torch.tensor(train_trial_idxs) == k
                ).reshape(-1) for k in train], 
                train_time_idxs = [torch.argwhere(
                    torch.tensor(train_time_idxs) == t
                ).reshape(-1) for t in range(n_t)],
                test_trial_idxs = [torch.argwhere(
                    torch.tensor(test_trial_idxs) == k
                ).reshape(-1) for k in test],
                test_time_idxs = [torch.argwhere(
                    torch.tensor(test_time_idxs) == t
                ).reshape(-1) for t in range(n_t)]
            )

            encoded_r, encoded_lam, encoded_mu, encoded_cov, elbos = cavi.encode(
                s = train_spike_features[:,1:],
                y = train_behaviors, 
                max_iter = cavi_max_iter
            )

            post_params = {
                "lambdas": encoded_lam.numpy(),
                "means": gmm.means_,
                "covs": gmm.covariances_,
            }

            mixture_weights, weight_matrix = compute_cavi_weight_matrix(
                bin_spike_features, y_train, y_pred, train, test, post_params
            )

    return weight_matrix

