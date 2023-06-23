import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
import torch
import torch.distributions as D



class ModelDataLoader():
    def __init__(
        self, 
        bin_spike_features, 
        bin_behaviors,
        bin_trial_idxs, 
        bin_time_idxs
    ):
        """
        Data loader for the ADVI / CAVI model. 
        
        Args:
            bin_spike_features: a nested list w/ the structure:
                                for each k:
                                    for each t:
                                        size (n_t_k, 1+n_d) array, n_d = spike feature dim
            bin_behaviors: size (n_k, n_t) array
            bin_trial_idxs: a list of trial index
            bin_time_idxs: a list of time bin index
        """
        
        self.bin_spike_features = bin_spike_features
        self.bin_behaviors = bin_behaviors
        self.bin_trial_idxs = bin_trial_idxs
        self.bin_time_idxs = bin_time_idxs
    
    def split_train_test(self, train, test):
        """Split the trials into train and test sets."""
        
        self.train_y = self.bin_behaviors[train]
        self.test_y = self.bin_behaviors[test]
        
        spike_features = np.concatenate(
            np.concatenate(self.bin_spike_features)
        )
        trial_idxs = np.concatenate(self.bin_trial_idxs)
        time_idxs = np.concatenate(self.bin_time_idxs)

        train_mask = np.sum(
            [trial_idxs == idx for idx in train], axis=0
        ).astype(bool)
        test_mask = np.sum(
            [trial_idxs == idx for idx in test], axis=0
        ).astype(bool)
        
        train_trial_idxs, test_trial_idxs = trial_idxs[train_mask], trial_idxs[test_mask]
        train_time_idxs, test_time_idxs = time_idxs[train_mask], time_idxs[test_mask]
        train_spike_features, test_spike_features = spike_features[train_mask], spike_features[test_mask]
        
        return train_spike_features, train_trial_idxs, train_time_idxs, \
               test_spike_features, test_trial_idxs, test_time_idxs
    


class ADVI(torch.nn.Module):
    def __init__(self, n_t, gmm, device):
        super().__init__()
        """
        ADVI model that handles both continuous and discrete behavioral correlates.
        
        Args:
            n_t: number of time bins in a trial 
            gmm: an instance of sklearn's Gaussian mixture model object
            device: the device (CPU or GPU) on which models are allocated
        """
        
        self.n_t = n_t
        self.n_c, self.n_d = gmm.means_.shape
        self.device = device
        
        # initialize parameters for variational distribution
        self.means = torch.nn.Parameter(torch.tensor(gmm.means_), requires_grad=False)
        self.covs = torch.nn.Parameter(torch.tensor(gmm.covariances_), requires_grad=False)
        
        # b ~ N(b_mu, exp(b_log_sig))
        self.b_mu = torch.nn.Parameter(torch.randn((self.n_c)))
        self.b_log_sig = torch.nn.Parameter(torch.randn((self.n_c)))
        
        # beta ~ N(beta_mu, exp(beta_log_sig))
        self.beta_mu = torch.nn.Parameter(torch.randn((self.n_c, self.n_t)))
        self.beta_log_sig = torch.nn.Parameter(torch.randn((self.n_c, self.n_t)))
        
        
    def _log_prior(self, b_sample, beta_sample):
        """
        Compute the log-likelihood of the prior. With continuous-valued model parameters, 
        we do not need to consider the jacobian term in the ADVI paper.
        """
        
        lp_b = D.Normal(torch.zeros((self.n_c)).to(self.device), 
                        torch.ones((self.n_c)).to(self.device)).log_prob(b_sample).sum()

        lp_beta = D.Normal(torch.zeros((self.n_c, self.n_t)).to(self.device), 
                           torch.ones((self.n_c, self.n_t)).to(self.device)).log_prob(beta_sample).sum()
        
        return lp_b + lp_beta
    
    
    def _log_q(self, b_sample, beta_sample):
        """
        Compute the log-likelihood of the variational distribution. 
        """
        
        lq_b = self.b.log_prob(b_sample).sum()

        lq_beta = self.beta.log_prob(beta_sample).sum()
        
        return lq_b + lq_beta
    
    
    def compute_elbo(
        self, 
        spike_features, 
        trial_idxs, 
        time_idxs, 
        model_params, 
        scaling_factor,
        fast_compute=True
    ):
        """
        Compute the evidence lower bound (ELBO).
        
        Args:
            spike_features: size (n_b, n_d) tensor,
                            n_b = number of spikes in a batch
                            n_d = spike feature dim
            trial_idxs: size (n_b,) tensor   
            time_idxs: size (n_b,) tensor 
            model_params: a dict of model parameters that contains b, beta, means and covs
            scaling_factor: factor to scale the ELBO for stochastic optimization with data subsampling
            fast_compute: whether to speed up the computation (only when batch_size = 1)
            
        Returns:
            elbo: float; ELBO
        """
        
        unique_trial_idxs = torch.unique(trial_idxs).int()
        n_k = len(unique_trial_idxs)
        
        elbo = self._log_prior(model_params["b"], model_params["beta"])
        elbo -= self._log_q(model_params["b"], model_params["beta"])

        if fast_compute:
            
            assert n_k == 1, "fast ELBO computation only works when batch_size = 1."
            
            mixing_props = torch.zeros((len(spike_features), self.n_c)).to(self.device)
            for k in range(n_k):
                for t in range(self.n_t):
                    trial_time_idx = torch.logical_and(
                        trial_idxs == unique_trial_idxs[k], time_idxs == t
                    )
                    mixing_props[trial_time_idx] = model_params["pi"][k,:,t]

            mix = D.Categorical(mixing_props)
            comp = D.MultivariateNormal(self.means, self.covs)
            gmm = D.MixtureSameFamily(mix, comp)
            elbo += gmm.log_prob(spike_features).sum() * scaling_factor
            
        else:
            for k in range(n_k):
                for t in range(self.n_t):
                    trial_time_idx = torch.logical_and(
                        trial_idxs == unique_trial_idxs[k], time_idxs == t
                    )
                    sub_spike_features = spike_features[trial_time_idx]
                    mix = D.Categorical(model_params["pi"][k,:,t])
                    comp = D.MultivariateNormal(self.means, self.covs)
                    gmm = D.MixtureSameFamily(mix, comp)
                    if len(sub_spike_features) > 0:
                        elbo += gmm.log_prob(sub_spike_features).sum() * scaling_factor
            
        return elbo


    def forward(self, behaviors):
        """
        Performs forward pass computation on the input tensors.
        
        Args:
            behaviors: size (n_k,) or (n_k, n_t) tensor    
            
        Returns:
            model_params: a dict of model parameters that contains b, beta, means and covs
        """
        
        n_k = len(behaviors)
        
        # define variational variables
        self.b = D.Normal(self.b_mu, self.b_log_sig.exp())
        self.beta = D.Normal(self.beta_mu, self.beta_log_sig.exp())
        
        # sample from variational distributions
        b_sample = self.b.rsample()
        beta_sample = self.beta.rsample()
                 
        # compute mixing proportions 
        log_lambdas = torch.zeros((n_k, self.n_c, self.n_t))
        log_lambdas = (b_sample[None,:,None] + beta_sample[None,:,:] * behaviors[:,None,:])
        log_pis = log_lambdas - torch.logsumexp(log_lambdas, 1)[:,None,:]
                   
        model_params = {
            "pi": log_pis.exp(), 
            "b": b_sample, 
            "beta": beta_sample, 
            "lambda": log_lambdas.exp()
        }
                                          
        return model_params
    
    
def train_advi(
    model, 
    spike_features, 
    behaviors, 
    trial_idxs, 
    time_idxs, 
    batch_idxs, 
    optim, 
    max_iter=1000,
    fast_compute=True,
    stochastic=True
):
    """
    Trains the ADVI model on the provided dataset.
    
    Args:
        spike_features: size (N, n_d) tensor,
                        N = number of spikes in train set
                        n_d = spike feature dim
        behaviors: size (n_k,) or (n_k, n_t) tensor 
        trial_idxs: size (N,) tensor 
        time_idxs: size (N,) tensor 
        batch_idxs: trial index allocated to each batch
        optim: pytorch optimizer to update the gradients 
        max_iter: maximum number of iterations  
        fast_compute: whether to speed up the computation (only when batch_size = 1)
        
    Returns:
        elbos: a list containing the computed ELBO
    """
    
    assert max_iter > 5, "need more iterations to train the model."
    N = len(torch.unique(trial_idxs))
    n_batches, batch_size = len(batch_idxs), len(batch_idxs[0])
    fast_compute = False if batch_size > 1 else True
    
    elbos = []
    for it in tqdm(range(max_iter), desc="Train ADVI"):
        
        if stochastic:
            
            idx = np.random.choice(range(n_batches), 1).item()
            batch_idx = batch_idxs[idx]
            mask = np.sum([trial_idxs[0] == idx for idx in batch_idx], axis=0)

            batch_spike_features = spike_features[mask]
            batch_behaviors = behaviors[list(batch_idx)]
            batch_trial_idxs = trial_idxs[mask]
            batch_time_idxs = time_idxs[mask]

            model_params = model(batch_behaviors)

            loss = - model.compute_elbo(
                batch_spike_features, 
                batch_trial_idxs, 
                batch_time_idxs, 
                model_params, 
                scaling_factor=batch_size/N,
                fast_compute=fast_compute
            )
            loss.backward()
            elbo = - loss.item()
            optim.step()
            optim.zero_grad()
            elbos.append(elbo)
            
        else:
            
            tot_elbo = 0
            for idx, batch_idx in enumerate(batch_idxs): 
                
                mask = np.sum([trial_idxs[0] == idx for idx in batch_idx], axis=0)
                
                batch_spike_features = spike_features[mask]
                batch_behaviors = behaviors[list(batch_idx)]
                batch_trial_idxs = trial_idxs[mask]
                batch_time_idxs = time_idxs[mask]
                
                model_params = model(batch_behaviors)
                
                loss = - model.compute_elbo(
                    batch_spike_features, 
                    batch_trial_idxs, 
                    batch_time_idxs, 
                    model_params, 
                    scaling_factor=batch_size/N,
                    fast_compute=fast_compute
                )
                
                loss.backward()
                tot_elbo -= loss.item()
                optim.step()
                optim.zero_grad()
            elbos.append(tot_elbo)
        
    elbos = [elbo for elbo in elbos]
    
    return elbos


def compute_posterior_weight_matrix(
    x, 
    y_train, 
    y_pred, 
    train, 
    test, 
    post_params,
    n_workers=4
):
    """
    Compute the posterior dynamic mixture weights for GMM and the posterior weight matrix 
    as input to the behavior decoder. (Parallel computing enabled)

    Args:
        x: a nested list w/ the structure:
           for each k:
               for each t:
                   size (n_t_k, 1+n_d) array, n_d = spike feature dim
        y_train (y_pred): size (n_k,) or (n_k, n_t) array
        train: trial index in the train set
        test: trial index in the test set
        post_params: a dict of model parameters that contains b, beta, means and covs 
        n_workers: number of workers in multiprocessing

    Returns:
        mixture_weights: size (n_k, n_c, n_t) array
        weight_matrix: size (n_k, n_c, n_t) array
    """
    
    align_idxs = np.append(train, test)
    if len(y_train.shape) == 1:
        y = np.hstack([y_train, y_pred])
    else:
        y = np.vstack([y_train, y_pred])
    x = [x[idx] for idx in align_idxs]
    
    n_k = len(y)
    n_c, n_t = post_params["beta"].shape
    
    match_idxs = [np.argwhere(np.array(align_idxs)==k).item() for k in range(n_k)]
        
    if n_workers == 1:
        
        log_lambdas = np.zeros((n_k, n_c, n_t))
        log_lambdas = (
            post_params["b"][:,None,None] + post_params["beta"][:,:,None] * y.T
        ).transpose((-1,0,1))
        log_pis = log_lambdas - logsumexp(log_lambdas, 1)[:,None,:]
        mixture_weights = np.exp(log_pis)

        weight_matrix = np.zeros((n_k, n_c, n_t))
        for k in tqdm(range(n_k), desc="Compute weight matrix"):
            for t in range(n_t):
                post_gmm = GaussianMixture(n_components=n_c, covariance_type='full')
                post_gmm.weights_ = mixture_weights[k,:,t]
                post_gmm.means_ = post_params["means"]
                post_gmm.covariances_ = post_params["covs"]
                post_gmm.precisions_cholesky_ = np.linalg.cholesky(
                    np.linalg.inv(post_params["covs"])
                )
                if len(x[k][t]) > 0:
                    weight_matrix[k,:,t] = post_gmm.predict_proba(x[k][t][:,1:]).sum(0)
    else:
        
        pool = multiprocessing.Pool(processes=n_workers)
        
        results = [pool.apply_async(
                    compute_weight_single_process, 
                    args=(x[k], y[k], post_params)
                    ) for k in range(n_k)]
        outputs = [result.get() for result in results]
        
        pool.close()
        pool.join()
    
        mixture_weights, weight_matrix = [], []
        for _, out in enumerate(outputs):
            mixture_weights.append(out[0])
            weight_matrix.append(out[1])
        mixture_weights, weight_matrix = np.vstack(mixture_weights), np.vstack(weight_matrix) 
          
    mixture_weights, weight_matrix = mixture_weights[match_idxs], weight_matrix[match_idxs]

    return mixture_weights, weight_matrix


def compute_weight_single_process(x, y, post_params):
    """
    Compute the posterior weight matrix for parallel computing.
    """
    
    y = y.reshape(1,-1)
    n_c, n_t = post_params["beta"].shape
    
    log_lambdas = np.zeros((1, n_c, n_t))
    log_lambdas = (post_params["b"][:,None,None] + \
                   post_params["beta"][:,:,None] * y.T).transpose((-1,0,1))
    log_pis = log_lambdas - logsumexp(log_lambdas, 1)[:,None,:]
    mixture_weights = np.exp(log_pis)

    weight_matrix = np.zeros((1, n_c, n_t))
    for t in range(n_t):
        post_gmm = GaussianMixture(n_components=n_c, covariance_type='full')
        post_gmm.weights_ = mixture_weights[:,:,t]
        post_gmm.means_ = post_params["means"]
        post_gmm.covariances_ = post_params["covs"]
        post_gmm.precisions_cholesky_ = np.linalg.cholesky(
            np.linalg.inv(post_params["covs"])
        )
        if len(x[t]) > 0:
            weight_matrix[:,:,t] = post_gmm.predict_proba(x[t][:,1:]).sum(0)
                
    return mixture_weights, weight_matrix
