import numpy as np
import torch
import torch.distributions as D
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

def safe_log(x, minval=1e-10):
    return torch.log(x + minval)

def safe_divide(x, y):
    return torch.clip(x / y, min = 0, max = 1)


class ADVI(torch.nn.Module):
    def __init__(self, n_k, n_t, n_c, n_d, init_means, init_covs):
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
        self.b_mu = torch.nn.Parameter(torch.randn((n_k)))
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

        lp_beta = D.Normal(torch.zeros((self.n_c, self.n_t)), torch.ones((self.n_c, self.n_t))).log_prob(beta_sample).sum()
        
        return lp_b + lp_beta
    
    def _log_q(self, b_sample, beta_sample):
        
        lq_b = self.b.log_prob(b_sample).sum()

        lq_beta = self.beta.log_prob(beta_sample).sum()
        
        return lq_b + lq_beta
        
    def forward(self, s, y, ks, ts, sampling=True):
        '''
        Inputs:
        -------
        s: (n, d) array; n = # of spikes in each batch, d = spike feature dimension.
        y: array of size (k,) if y is binary, and of size (k, t) if y is continuous; 
           k = # of trials in each batch, t = # of time bins. 
        ks: (n,) index array that denotes the trial each spike belongs to.  
        ts: (n,) index array that denotes the time bin each spike falls into. 
        sampling: if True then sample from the variational q; if False then use the means of q. 
        '''
        
        # define global variational variables
        self.b = D.Normal(self.b_mu, self.b_log_sig.exp())
        self.beta = D.Normal(self.beta_mu, self.beta_log_sig.exp())
        
        
        # sample from variational distributions
        if sampling:
            b_sample = self.b.rsample()
            beta_sample = self.beta.rsample()
        else:
            b_sample = self.b.loc
            beta_sample = self.beta.loc
                 
                
        # compute mixing proportions 
        log_lambdas = torch.zeros((self.n_k, self.n_c, self.n_t))
        for k in range(self.n_k):
            for t in range(self.n_t):
                if len(y.shape) == 1:
                    log_lambdas[k,:,t] = b_sample + beta_sample[:,t] * y[k]
                else:
                    log_lambdas[k,:,t] = b_sample + beta_sample[:,t] * y[k][t]
        log_pis = log_lambdas - torch.logsumexp(log_lambdas, 1)[:,None,:]
                                          
                                          
        # compute log-likelihood
        ll = torch.zeros((s.shape[0], self.n_c))
        for j in range(self.n_c):
            ll[:,j] = D.multivariate_normal.MultivariateNormal(
                            loc=self.means[j], 
                            covariance_matrix=self.covs[j]
                        ).log_prob(s)
            
        
        # compute local variational variables
        r = torch.zeros((s.shape[0], self.n_c))
        for k in range(self.n_k):
            for t in range(self.n_t):
                k_t_idx = torch.logical_and(ks==torch.unique(ks).int()[k], ts==t)
                r[k_t_idx] = torch.exp(ll[k_t_idx] + log_pis[k,:,t])
                r[k_t_idx] = r[k_t_idx]/r[k_t_idx].sum(1)[:,None]
                            
                                          
        # compute ELBO
        elbo = self._log_prior_plus_logabsdet_J(b_sample, beta_sample)
        elbo -= self._log_q(b_sample, beta_sample) 
        for k in range(self.n_k):
            for t in range(self.n_t):
                k_t_idx = torch.logical_and(ks==torch.unique(ks).int()[k], ts==t)
                elbo += torch.sum(r[k_t_idx] * (ll[k_t_idx] + log_pis[k,:,t]))
                elbo -= torch.sum(r[k_t_idx] * safe_log(r[k_t_idx]))
        
        return elbo
    
    def train(self, s, y, ks, ts, batch_ids, optim, max_iter):
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
                mask = torch.logical_and(ks >= batch_idx[0], ks <= batch_idx[-1])
                batch_s = s[mask]
                batch_y = y[list(batch_idx)]
                batch_ks = ks[mask]
                batch_ts = ts[mask]
                loss = - self(batch_s, batch_y, batch_ks, batch_ts) / N
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
    
    def calc_dynamic_mixing_proportions(self, y):
        n_k = len(y)
        log_lambdas = torch.zeros((n_k, self.n_c, self.n_t))
        for k in range(n_k):
            for t in range(self.n_t):
                if len(y.shape) == 1:
                    log_lambdas[k,:,t] = self.b.loc + self.beta.loc[:,t] * y[k]
                else:
                    log_lambdas[k,:,t] = self.b.loc + self.beta.loc[:,t] * y[k][t]
        log_pis = log_lambdas - torch.logsumexp(log_lambdas, 1)[:,None,:]
        return log_pis.exp().detach().numpy()
    
    def encode_gmm(self, data, train, test, y_train, y_pred):
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
        n_k = len(train) + len(test)
        log_lambdas = np.zeros((n_k, self.n_c, self.n_t))
        for i, k in enumerate(train):
            for t in range(self.n_t):
                if len(y_train.shape) == 1:
                    log_lambdas[k,:,t] = self.b.loc.detach().numpy() + \
                                         self.beta.loc[:,t].detach().numpy() * y_train[i]
                else:
                    log_lambdas[k,:,t] = self.b.loc.detach().numpy() + \
                                     self.beta.loc[:,t].detach().numpy() * y_train[i][t]

        for i, k in enumerate(test):
            for t in range(self.n_t):
                if len(y_pred.shape) == 1:
                    log_lambdas[k,:,t] = self.b.loc.detach().numpy() + \
                                         self.beta.loc[:,t].detach().numpy() * y_pred[i]
                else:
                    log_lambdas[k,:,t] = self.b.loc.detach().numpy() + \
                                     self.beta.loc[:,t].detach().numpy() * y_pred[i][t]

        log_pis = log_lambdas - logsumexp(log_lambdas, 1)[:,None,:]
        encoded_pis = np.exp(log_pis)
        
        encoded_weights = np.zeros_like(log_lambdas)
        for k in range(encoded_weights.shape[0]):
            for t in range(self.n_t):
                encoded_gmm = GaussianMixture(n_components=self.n_c, covariance_type='full')
                encoded_gmm.weights_ = encoded_pis[k,:,t]
                encoded_gmm.means_ = self.means
                encoded_gmm.covariances_ = self.covs
                encoded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.covs))
                if len(data[k][t]) > 0:
                    encoded_weights[k,:,t] = encoded_gmm.predict_proba(data[k][t][:,1:]).sum(0)
                    
        return encoded_pis, encoded_weights
    
    
