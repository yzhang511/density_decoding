import numpy as np
import random
import torch
import torch.distributions as D
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, roc_auc_score


# helper functions
def safe_log(x, minval=1e-10):
    return torch.log(x + minval)

def safe_divide(x, y):
    return torch.clip(x / y, min = 0, max = 1)


# encoder for continuous behavior
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
        lp_b = D.Normal(torch.zeros((self.Nc)), torch.ones((self.Nc))).log_prob(b_sample).sum()

        # log prior sig + log jacobian
        lp_beta = D.Normal(torch.zeros((self.Nc, self.Nt)), torch.ones((self.Nc, self.Nt))).log_prob(beta_sample).sum()
        
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
    
    
    def train_advi(self, s, y, ks, ts, batch_ids, optim, max_iter):
        '''
        
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
                if (n+1) % 100 == 0:
                    print(f'iter: {i+1} batch {n+1}')
                optim.step()
                optim.zero_grad()
            print(f'iter: {i+1} total elbo: {tot_elbo:.2f}')
            elbos.append(tot_elbo)
        elbos = [elbo for elbo in elbos]
        return elbos
    
    
    def calc_dynamic_mixing_proportions(self, y):
        '''
        
        '''
        log_lambdas = torch.zeros((len(y), self.Nc, self.Nt))
        for k in range(len(y)):
            for t in range(self.Nt):
                log_lambdas[k,:,t] = self.b.loc + self.beta.loc[:,t] * y[k][t]

        log_pis = log_lambdas - torch.logsumexp(log_lambdas, 1)[:,None,:]
        return log_pis.exp().detach().numpy()
    
    
    def encode_gmm(self, data, train, test, y_train, y_pred):
        
        Nk = len(train) + len(test)
        log_lambdas = np.zeros((Nk, self.Nc, self.Nt))
        for i, k in enumerate(train):
            for t in range(self.Nt):
                log_lambdas[k,:,t] = self.b.loc.detach().numpy() + \
                                     self.beta.loc[:,t].detach().numpy() * y_train[i][t]

        for i, k in enumerate(test):
            for t in range(self.Nt):
                log_lambdas[k,:,t] = self.b.loc.detach().numpy() + \
                                     self.beta.loc[:,t].detach().numpy() * y_pred[i][t]

        log_pis = log_lambdas - logsumexp(log_lambdas, 1)[:,None,:]
        encoded_pis = np.exp(log_pis)
        
        encoded_weights = np.zeros_like(log_lambdas)
        for k in range(encoded_weights.shape[0]):
            for t in range(self.Nt):
                encoded_gmm = GaussianMixture(n_components=self.Nc, covariance_type='full')
                encoded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.covs))
                encoded_gmm.weights_ = encoded_pis[k,:,t]
                encoded_gmm.means_ = self.means
                encoded_gmm.covariances_ = self.covs
                if len(data[k][t]) > 0:
                    encoded_weights[k,:,t] = encoded_gmm.predict_proba(data[k][t][:,1:]).sum(0)
        return encoded_pis, encoded_weights
    
    

class CAVI():
    def __init__(self, init_mu, init_cov, init_lam, 
                 train_k_ids, train_t_ids, test_k_ids, test_t_ids):
        
        self.train_K = len(train_k_ids)
        self.test_K = len(test_k_ids)
        self.T = len(train_t_ids)
        self.C = init_mu.shape[0]
        self.D = init_mu.shape[1]
        self.init_mu = init_mu
        self.init_cov = init_cov
        self.init_lam = init_lam
        self.train_k_ids = train_k_ids
        self.train_t_ids = train_t_ids
        self.test_k_ids = test_k_ids
        self.test_t_ids = test_t_ids
        
    
    def _compute_normal_log_dens(self, s, mu, cov):
        
        log_dens = []
        for j in range(self.C):
            log_dens.append(
                torch.tensor(
                    multivariate_normal.logpdf(s, mu[j], cov[j])
                )
            )
        return torch.vstack(log_dens).T # (*, C)
    
    
    def _compute_enc_elbo(self, r, y, log_dens, norm_lam):
        
        elbo_1 = torch.sum(torch.tensor(
            [torch.einsum('i,i->', r[:,j], log_dens[:,j]) for j in range(self.C)]
        ))

        elbo_2 = torch.tensor(
            [ torch.einsum('ij,il,j->', r[self.train_t_ids[t]], 
                           y[self.train_t_ids[t]], norm_lam[:,t,1]) +
              torch.einsum('ij,il,j->', r[self.train_t_ids[t]], 
                           1-y[self.train_t_ids[t]], norm_lam[:,t,0]) for t in range(self.T) ]
            ).sum()

        elbo_3 = - torch.einsum('ij,ij->', safe_log(r), r)

        elbo = elbo_1 + elbo_2 + elbo_3 

        return elbo
    
    
    def _compute_dec_elbo(self, r, log_dens, norm_lam, nu, nu_k, p):
        
        elbo_1 = torch.sum(torch.tensor(
            [torch.einsum('i,i->', r[:,j], log_dens[:,j]) for j in range(self.C)]
        ))

        elbo_2 = torch.tensor(
            [ torch.einsum('ij,il,j->', r[self.test_t_ids[t]], 
                           nu[self.test_t_ids[t]], norm_lam[:,t,1]) +
              torch.einsum('ij,il,j->', r[self.test_t_ids[t]], 
                           1-nu[self.test_t_ids[t]], norm_lam[:,t,0]) for t in range(self.T) ]
            ).sum()

        elbo_3 = torch.sum(nu_k * safe_log(p) + (1-nu_k) * safe_log(1-p))

        elbo_4 = - torch.einsum('ij,ij->', safe_log(r), r)

        elbo_5 = - torch.sum(safe_log(nu_k) * nu_k)

        elbo = elbo_1 + elbo_2 + elbo_3 + elbo_4 + elbo_5

        return elbo
    
    
    def _encode_e_step(self, r, y, log_dens, norm_lam):
        
        for t in range(self.T):
            r[self.train_t_ids[t]] = torch.exp( log_dens[self.train_t_ids[t]] + \
                      torch.einsum('il,j->ij', y[self.train_t_ids[t]], norm_lam[:,t,1]) + \
                      torch.einsum('il,j->ij', 1-y[self.train_t_ids[t]], norm_lam[:,t,0])
            )
            r[self.train_t_ids[t]] = torch.einsum('ij,i->ij', r[self.train_t_ids[t]], 1/r[self.train_t_ids[t]].sum(1))
        return r
        
    
    def _encode_m_step(self, s, r, y, mu, lam):
        
        for j in range(self.C):
            no_j_idx = torch.cat([torch.arange(j), torch.arange(j+1, self.C)])
            lam_sum_no_j = lam[no_j_idx,:,:].sum(0)
            for t in range(self.T):
                num1 = torch.einsum('i,il,->', r[self.train_t_ids[t],j], y[self.train_t_ids[t]], lam_sum_no_j[t,1])
                denom1 = np.einsum('ij,il->', r[self.train_t_ids[t]][:,no_j_idx], y[self.train_t_ids[t]])
                num0 = torch.einsum('i,il,->', r[self.train_t_ids[t],j], 1-y[self.train_t_ids[t]], lam_sum_no_j[t,0])
                denom0 = np.einsum('ij,il->', r[self.train_t_ids[t]][:,no_j_idx], 1-y[self.train_t_ids[t]])
                lam[j,t,1], lam[j,t,0] = num1 / denom1, num0 / denom0
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))

        norm = r.sum(0)
        mu = torch.einsum('j,ij,ip->jp', 1/norm, r, s)
        cov = [torch.einsum(
            ',i,ip,id->pd', 1/norm[j], r[:,j], s-mu[j], s-mu[j] ) for j in range(self.C)]
        
        return mu, cov, lam, norm_lam
    
    
    def _decode_e_step(self, r, log_dens, norm_lam, nu, nu_k, p):
        
        for t in range(self.T):
            r[self.test_t_ids[t]] = torch.exp( log_dens[self.test_t_ids[t]] + \
                      torch.einsum('il,j->ij', nu[self.test_t_ids[t]], norm_lam[:,t,1]) + \
                      torch.einsum('il,j->ij', 1-nu[self.test_t_ids[t]], norm_lam[:,t,0])
            )
            r[self.test_t_ids[t]] = torch.einsum('ij,i->ij', r[self.test_t_ids[t]], 1/r[self.test_t_ids[t]].sum(1))
        
        for k in range(self.test_K):
            y_tilde0, y_tilde1 = safe_log(1-p), safe_log(p)
            for t in range(self.T):
                k_t_ids = np.intersect1d(self.test_k_ids[k], self.test_t_ids[t])
                y_tilde0 += torch.einsum('ij,j->', r[k_t_ids], norm_lam[:,t,0])
                y_tilde1 += torch.einsum('ij,j->', r[k_t_ids], norm_lam[:,t,1])
            # y_tilde explode to 0 after exp(); need offset to ensure y_tilde stay in range
            offset = 1. / (torch.min(torch.tensor([y_tilde0, y_tilde1])) / -745.) 
            y_tilde0, y_tilde1 = torch.exp(y_tilde0 * offset), torch.exp(y_tilde1 * offset)
            nu_k[k] = safe_divide(y_tilde1, y_tilde0 + y_tilde1)
            nu[self.test_k_ids[k]] = nu_k[k]
            
        return r, nu, nu_k
    
    
    def _decode_m_step(self, s, r, nu_k, mu):
        
        p = nu_k.sum() / self.test_K
    
        norm = r.sum(0)
        mu = torch.einsum('j,ij,ip->jp', 1/norm, r, s)
        cov = [torch.einsum(
            ',i,ip,id->pd', 1/norm[j], r[:,j], s-mu[j], s-mu[j]) for j in range(self.C)]
        
        return p, mu, cov
    
    
    def encode(self, s, y, max_iter=20, eps=1e-6):
        # initialize
        r = torch.ones((s.shape[0], self.C)) / self.C
        lam = self.init_lam.clone()
        mu, cov = self.init_mu.clone(), self.init_cov.clone()
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))
        
        # compute initial elbo
        log_dens = self._compute_normal_log_dens(s, mu, cov)
        elbo = self._compute_enc_elbo(r, y, log_dens, norm_lam)
        convergence = 1.
        elbos = [elbo]
        print(f'initial elbo: {elbos[-1]:.2f}')
        
        it = 1
        while convergence > eps or convergence < 0: 
            # E step
            r = self._encode_e_step(r, y, log_dens, norm_lam)
            
            # M step
            mu, cov, lam, norm_lam = self._encode_m_step(s, r, y, mu, lam)
            
            # compute new elbo
            log_dens = self._compute_normal_log_dens(s, mu, cov)
            elbo = self._compute_enc_elbo(r, y, log_dens, norm_lam)
            elbos.append(elbo)
            convergence = elbos[-1] - elbos[-2]

            print(f'iter: {it} elbo: {elbos[-1]:.2f}.')
            it +=1 
            if it > max_iter: 
                print('reached max iter allowed.')
                break

        if abs(convergence) <= eps:
            print('converged.')
            
        return r, lam, mu, torch.stack(cov), elbos
    
    
    def decode(self, s, init_p, init_mu, init_cov, init_lam, 
                test_k_ids, test_ids, max_iter=20, eps=1e-6):
        # initialize
        p = init_p.clone()
        r = torch.ones((s.shape[0], self.C)) / self.C
        mu, cov = init_mu.clone(), init_cov.clone()
        lam = init_lam.clone()
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))
        nu_k = torch.rand(self.test_K)
        nu = torch.zeros(s.shape[0])
        for k in range(self.test_K):
            nu[test_k_ids == test_ids[k]] = nu_k[k]
        nu = nu.reshape(-1,1)
        
        # compute initial elbo
        log_dens = self._compute_normal_log_dens(s, mu, cov)
        elbo = self._compute_dec_elbo(r, log_dens, norm_lam, nu, nu_k, p)
        convergence = 1.
        elbos = [elbo]
        print(f'initial elbo: {elbos[-1]:.2f}')
        
        it = 1
        while convergence > eps or convergence < 0:
            # E step
            r, nu, nu_k = self._decode_e_step(r, log_dens, norm_lam, nu, nu_k, p)
            
            # M step
            p, mu, cov = self._decode_m_step(s, r, nu_k, mu)
            
            # compute new elbo
            log_dens = self._compute_normal_log_dens(s, mu, cov)
            elbo = self._compute_dec_elbo(r, log_dens, norm_lam, nu, nu_k, p)
            elbos.append(elbo)
            convergence = elbos[-1] - elbos[-2]

            print(f'iter: {it} elbo: {elbos[-1]:.2f}.')
            it +=1 
            if it > max_iter: 
                print('reached max iter allowed.')
                break

        if abs(convergence) <= eps:
            print('converged.')
    
        return r, nu_k, mu, torch.stack(cov), p, elbos
    
    
    def eval_performance(self, nu_k, y_test):
        acc = accuracy_score(y_test, 1. * ( nu_k > .5 ))
        auc = roc_auc_score(y_test, nu_k)
        print(f'decoding accuracy is {acc:.2f}')
        print(f'decoding auc is {auc:.2f}')
        return acc, auc
    
    
    
    def encode_gmm(self, trials, lams, means, covs, train, test, y_train, y_hat):
        
        enc_pis = lams / lams.sum(0)
        
        enc_all = np.zeros((len(train) + len(test), self.C, self.T))
        for i, k in enumerate(train):
            for t in range(self.T):
                enc_gmm = GaussianMixture(n_components=self.C, covariance_type='full')
                enc_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
                enc_gmm.weights_ = enc_pis[:,t,y_train[i]]
                enc_gmm.means_ = means
                enc_gmm.covariances_ = covs
                if len(trials[k][t]) > 0:
                    enc_all[k,:,t] = enc_gmm.predict_proba(trials[k][t][:,1:]).sum(0)
                    
        for i, k in enumerate(test):
            for t in range(self.T):
                enc_gmm = GaussianMixture(n_components=self.C, covariance_type='full')
                enc_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
                enc_gmm.weights_ = enc_pis[:,t,y_hat[i]]
                enc_gmm.means_ = means
                enc_gmm.covariances_ = covs
                if len(trials[k][t]) > 0:
                    enc_all[k,:,t] = enc_gmm.predict_proba(trials[k][t][:,1:]).sum(0)
        
        return enc_pis, enc_all
        
