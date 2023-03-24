import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

# helper functions
def safe_log(x, minval=1e-10):
    return torch.log(x + minval)


class ADVI(torch.nn.Module):

    def __init__(self, Nk, Nt, Nc, Nd, init_means, init_covs):
        super(ADVI, self).__init__()
        self.Nk = Nk
        self.Nt = Nt
        self.Nc = Nc
        self.Nd = Nd
        
        # initialize variables for variational distribution
        self.means = torch.nn.Parameter(torch.tensor(init_means), requires_grad=False)
        self.covs = torch.nn.Parameter(torch.tensor(init_covs), requires_grad=False)
        self.b_mu = torch.nn.Parameter(torch.randn((Nc)))
        self.b_log_sig = torch.nn.Parameter(torch.randn((Nc)))
        self.beta_mu = torch.nn.Parameter(torch.randn((Nc, Nt)))
        self.beta_log_sig = torch.nn.Parameter(torch.randn((Nc, Nt)))
        
    def log_prior_plus_logabsdet_J(self, b_sample, beta_sample):
        
        # log prior for beta, evaluated at sampled values for beta
        lp_b = D.Normal(torch.zeros((Nc)), torch.ones((Nc))).log_prob(b_sample).sum()

        # log prior sig + log jacobian
        lp_beta = D.Normal(torch.zeros((Nc, Nt)), torch.ones((Nc, Nt))).log_prob(beta_sample).sum()
        
        return lp_b + lp_beta
    
    
    def log_q(self, b_sample, beta_sample):
        
        lq_b = self.b.log_prob(b_sample).sum()

        lq_beta = self.beta.log_prob(beta_sample).sum()
        
        return lq_b + lq_beta
        
        
    def forward(self, s, y, ks, ts, sampling=True):
        
        # define global variational parameters
        self.b = D.Normal(self.b_mu, self.b_log_sig.exp())
        self.beta = D.Normal(self.beta_mu, self.beta_log_sig.exp())
        
        # sample from the variational distributions
        if sampling:
            b_sample = self.b.rsample()
            beta_sample = self.beta.rsample()
        else:
            b_sample = self.b.loc
            beta_sample = self.beta.loc
                    
        # mixing proportions 
        log_lambdas = torch.zeros((self.Nk, self.Nc, self.Nt))
        for k in range(self.Nk):
            for t in range(self.Nt):
                log_lambdas[k,:,t] = b_sample + beta_sample[:,t] * y[k][t]
        log_pis = log_lambdas - torch.logsumexp(log_lambdas, 1)[:,None,:]
                                          
                                          
        # compute log-likelihood
        ll = torch.zeros((s.shape[0], self.Nc))
        for j in range(self.Nc):
            ll[:,j] = D.multivariate_normal.MultivariateNormal(
                            loc=self.means[j], 
                            covariance_matrix=self.covs[j]
                        ).log_prob(s)
            
        
        # compute local variational parameters
        r = torch.zeros((s.shape[0], self.Nc))
        for k in range(self.Nk):
            for t in range(self.Nt):
                k_t_idx = torch.logical_and(ks == torch.unique(ks).int()[k], ts == t)
                r[k_t_idx] = torch.exp( ll[k_t_idx] + log_pis[k,:,t] )
                r[k_t_idx] = r[k_t_idx] / r[k_t_idx].sum(1)[:,None]
                            
                                          
        # compute ELBO
        elbo = 0
        for k in range(self.Nk):
            for t in range(self.Nt):
                k_t_idx = torch.logical_and(ks == torch.unique(ks).int()[k], ts == t)
                elbo += torch.sum( r[k_t_idx] * ll[k_t_idx] )
                elbo += torch.sum( r[k_t_idx] * log_pis[k,:,t] )
                elbo -= torch.sum( r[k_t_idx] * safe_log(r[k_t_idx]) )
                
        elbo += self.log_prior_plus_logabsdet_J(b_sample, beta_sample)
        elbo -= self.log_q(b_sample, beta_sample)                                  
        
        return elbo
    
    
    def encode_gmm(self, trials, train, test, y_train, y_hat):
        
        Nk = len(train) + len(test)
        log_lambdas_hat = np.zeros((Nk, self.Nc, self.Nt))
        for i, k in enumerate(train):
            for t in range(self.Nt):
                log_lambdas_hat[k,:,t] = self.b.loc.detach().numpy() + \
                                         self.beta.loc[:,t].detach().numpy() * y_train[i][t]

        for i, k in enumerate(test):
            for t in range(self.Nt):
                log_lambdas_hat[k,:,t] = self.b.loc.detach().numpy() + \
                                         self.betas.loc[:,t].detach().numpy() * y_hat[i][t]

        log_pis_hat = log_lambdas_hat - logsumexp(log_lambdas_hat, 1)[:,None,:]
        enc_pis = np.exp(log_pis_hat)
        
        enc_all = np.zeros((Nk, self.Nc, self.Nt))
        for k in range(enc_all.shape[0]):
            for t in range(self.Nt):
                enc_gmm = GaussianMixture(n_components=self.Nc, covariance_type='full')
                enc_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.covs))
                enc_gmm.weights_ = enc_pis[k,:,t]
                enc_gmm.means_ = self.means
                enc_gmm.covariances_ = self.covs
                if len(trials[k][t]) > 0:
                    enc_all[k,:,t] = enc_gmm.predict_proba(trials[k][t][:,1:]).sum(0)
        
        return enc_pis, enc_all