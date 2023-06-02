import numpy as np
import torch
import torch.distributions as D
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

def safe_log(x, minval=1e-10):
    return torch.log(x + minval)


class ADVI(torch.nn.Module):
    def __init__(self, n_k, n_t, n_c, n_d, init_means, init_covs):
        super(ADVI, self).__init__()
        '''
        ADVI can be used for both the continuous and discrete behavior variables.
        
        Inputs:
        -------
        n_k: # of trials.
        n_t: # of time bins.
        n_c: # of gmm components.
        n_d: spike feature dimension. 
        '''
        self.n_k = n_k
        self.n_t = n_t
        self.n_c = n_c
        self.n_d = n_d
        
        # initialize parameters for variational distribution
        self.means = torch.nn.Parameter(torch.tensor(init_means), requires_grad=False)
        self.covs = torch.nn.Parameter(torch.tensor(init_covs), requires_grad=False)
        
        # b ~ N(b_mu, exp(b_log_sig))
        self.b_mu = torch.nn.Parameter(torch.randn((n_c)))
        self.b_log_sig = torch.nn.Parameter(torch.randn((n_c)))
        
        # beta ~ N(beta_mu, exp(beta_log_sig))
        self.beta_mu = torch.nn.Parameter(torch.randn((n_c, n_t)))
        self.beta_log_sig = torch.nn.Parameter(torch.randn((n_c, n_t)))
        
    def _log_prior_plus_logabsdet_J(self, b_sample, beta_sample):
        '''
        Since both b and beta are continuous-valued, we do not need to consider the jacobian term
        as typically done in ADVI. 
        '''
        # log prior for b and beta, evaluated at sampled values 
        lp_b = D.Normal(torch.zeros((self.n_c)), torch.ones((self.n_c))).log_prob(b_sample).sum()

        lp_beta = D.Normal(torch.zeros((self.n_c, self.n_t)), 
                           torch.ones((self.n_c, self.n_t))).log_prob(beta_sample).sum()
        
        return lp_b + lp_beta
    
    def _log_q(self, b_sample, beta_sample):
        
        lq_b = self.b.log_prob(b_sample).sum()

        lq_beta = self.beta.log_prob(beta_sample).sum()
        
        return lq_b + lq_beta
    
    
    def compute_elbo(self, s, ks, ts, model_params):
        elbo = self._log_prior_plus_logabsdet_J(model_params["b"], model_params["beta"])
        elbo -= self._log_q(model_params["b"], model_params["beta"])

        for k in range(self.n_k):
            for t in range(self.n_t):
                k_t_idx = torch.logical_and(ks==torch.unique(ks).int()[k], ts==t)
                mix = D.Categorical(model_params["pi"][k,:,t])
                comp = D.MultivariateNormal(self.means, self.covs)
                gmm = D.MixtureSameFamily(mix, comp)
                if len(s[k_t_idx]) > 0:
                    elbo += gmm.log_prob(s[k_t_idx]).sum()
        return elbo
    
        
    def forward(self, s, y, ks, ts):
        '''
        Inputs:
        -------
        s: (n, d) array; n = # of spikes in each batch, d = spike feature dimension.
        y: array of size (k,) if y is binary, and of size (k, t) if y is continuous; 
           k = # of trials in each batch, t = # of time bins. 
        ks: (n,) index array that denotes the trial each spike belongs to.  
        ts: (n,) index array that denotes the time bin each spike falls into. 
        '''
        
        # define global variational variables
        self.b = D.Normal(self.b_mu, self.b_log_sig.exp())
        self.beta = D.Normal(self.beta_mu, self.beta_log_sig.exp())
        
        # sample from variational distributions
        b_sample = self.b.rsample()
        beta_sample = self.beta.rsample()
                 
        # compute mixing proportions 
        n_k = len(y)
        log_lambdas = torch.zeros((n_k, self.n_c, self.n_t))
        for k in range(n_k):
            for t in range(self.n_t):
                if len(y.shape) == 1:
                    log_lambdas[k,:,t] = b_sample + beta_sample[:,t] * y[k]
                else:
                    log_lambdas[k,:,t] = b_sample + beta_sample[:,t] * y[k][t]
        log_pis = log_lambdas - torch.logsumexp(log_lambdas, 1)[:,None,:]
                   
        model_params = {"lambda": log_lambdas.exp(), "pi": log_pis.exp(), "b": b_sample, "beta": beta_sample}
                                          
        return model_params
    
    
def train_advi(advi, s, y, ks, ts, batch_ids, optim, max_iter):
    '''
    Inputs:
    -------
    s: (n, d) array; n = # of spikes in each recording session.
    y: array of size (k,) if y is binary, and of size (k, t) if y is continuous; 
       k = # of trials in each session. 
    ks: (n,) index array that denotes the trial each spike belongs to.  
    ts: (n,) index array that denotes the time bin each spike falls into. 
    batch_ids: trials indices in each batch. 
    optim: pytorch optimizer. 
    max_iter: max iteration allowed. 
    '''
    elbos = []
    N = s.shape[0]
    for i in range(max_iter):
        tot_elbo = 0
        for n, batch_idx in enumerate(batch_ids): 
            mask = torch.logical_and(ks >= np.min(batch_idx), ks <= np.max(batch_idx))
            batch_s = s[mask]
            batch_y = y[list(batch_idx)]
            batch_ks = ks[mask]
            batch_ts = ts[mask]
            model_params = advi(batch_s, batch_y, batch_ks, batch_ts)
            loss = - advi.compute_elbo(batch_s, batch_ks, batch_ts, model_params) / N
            loss.backward()
            tot_elbo -= loss.item()
            # if (n+1) % 100 == 0:
            #     print(f'iter: {i+1} batch {n+1}')
            optim.step()
            optim.zero_grad()
        print(f'iter: {i+1} total elbo: {tot_elbo:.2f}')
        elbos.append(tot_elbo)
    elbos = [elbo for elbo in elbos]
    return elbos


def encode_gmm(advi, data, train, test, y_train, y_pred):
    '''
    Encoding the gmm with the dynamic mixing propotions. 

    Inputs:
    -------
    data: a nested list of spike features; data[k][t] contains spikes that fall into k-th trial and
          t-th time bin. data[k][t] is a (spike channels, spike features) array.
    train: training trial indices. 
    test:  test trial indices. 
    y_train: discrete or continuous behaviors in the training trials. 
    y_pred:  initially predicted behaviors in the test trials; can obtain using either
             multi-unit thresholding or vanilla gmm (with fixed mixing proportions).

    Outputs:
    -------
    encoded_pis: dynamic mixing proportions; (k, c, t) array. 
    encoded_weights: posterior assignment weight matrix from the encoded gmm; (k, c, t) array. 
    '''
    trial_idx = np.append(train, test)
    n_k = len(trial_idx) 
    if len(y_train.shape) == 1:
        y = np.hstack([y_train, y_pred])
    else:
        y = np.vstack([y_train, y_pred])
    
    # compute dynamic mixing proportions
    log_lambdas = np.zeros((n_k, advi.n_c, advi.n_t))
    for i, k in enumerate(trial_idx):
        for t in range(advi.n_t):
            if len(y.shape) == 1:
                log_lambdas[k,:,t] = advi.b.loc.detach().numpy() + \
                                     advi.beta.loc[:,t].detach().numpy() * y[i]
            else:
                log_lambdas[k,:,t] = advi.b.loc.detach().numpy() + \
                                     advi.beta.loc[:,t].detach().numpy() * y[i][t]
    log_pis = log_lambdas - logsumexp(log_lambdas, 1)[:,None,:]
    encoded_pis = np.exp(log_pis)

    # compute assignment weight matrix for decoding
    encoded_weights = np.zeros_like(log_lambdas)
    for k in range(encoded_weights.shape[0]):
        for t in range(advi.n_t):
            encoded_gmm = GaussianMixture(n_components=advi.n_c, covariance_type='full')
            encoded_gmm.weights_ = encoded_pis[k,:,t]
            encoded_gmm.means_ = advi.means
            encoded_gmm.covariances_ = advi.covs
            encoded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(advi.covs))
            if len(data[k][t]) > 0:
                encoded_weights[k,:,t] = encoded_gmm.predict_proba(data[k][t][:,1:]).sum(0)

    return encoded_pis, encoded_weights



