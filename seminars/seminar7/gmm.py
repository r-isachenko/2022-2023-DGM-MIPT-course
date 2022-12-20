import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as TD
import numpy as np

class GMM(nn.Module):

    @staticmethod
    def positive2logits(x):
        # inverse softplus transform
        # return x + torch.log(-torch.expm1(-x))
        return torch.log(x)

    @staticmethod
    def logits2positive(x):
        return torch.exp(x)
        # return F.softplus(x)

    @property
    def _L(self):
        L = torch.diag_embed(
            self.logits2positive(self._L_diag_logits), 
            offset=0, dim1=-2, dim2=-1)
        L_lower_ids = torch.tril_indices(self.dim, self.dim, -1)
        L[:, L_lower_ids[0], L_lower_ids[1]] = self._L_lower_logits
        return L

    @_L.setter
    def _L(self, L):
        assert len(L.shape) == 3
        L_lower_ids = torch.tril_indices(self.dim, self.dim, -1)
        L_lower_logits = L[:, L_lower_ids[0], L_lower_ids[1]]
        L_diag_logits = self.positive2logits(
            torch.diagonal(L, offset=0, dim1=-2, dim2=-1))
        self._L_lower_logits = nn.Parameter(L_lower_logits)
        self._L_diag_logits = nn.Parameter(L_diag_logits)

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, val):
        self._mu = nn.Parameter(val)

    @property
    def sigma(self):
        L = self._L
        return L @ L.transpose(1, 2)

    @sigma.setter
    def sigma(self, val):
        assert len(val.shape) == 3
        self._L = torch.cholesky(val)

    @property
    def pi(self):
        return F.softmax(self._log_pi)

    @pi.setter
    def pi(self, val):
        assert val.min() > 0.
        val = val/torch.sum(val)
        self.register_buffer('_log_pi', torch.log(val))
        # self._log_pi = torch.log(val)
        self._log_pi = nn.Parameter(torch.log(val))

    @property
    def gmm(self):
        mix = TD.Categorical(self.pi)
        mv_normals = TD.MultivariateNormal(self.mu, scale_tril=self._L)
        gmm = TD.MixtureSameFamily(mix, mv_normals)
        return gmm

    def __init__(self, K, dim, mu=None, sigma=None, pi=None):
        '''
        Define a model with known number of clusters and dimensions.
        :Parameters:
            - K: Number of Gaussian clusters
            - dim: Dimension 
            - mu: means of clusters (K, dim)
                       (default) random from uniform[-10, 10]
            - sigma: covariance matrices of clusters (K, dim, dim)
                          (default) Identity matrix for each cluster
            - pi: cluster weights (K,)
                       (default) equal value to all cluster i.e. 1/K
        '''
        super().__init__()
        self.K = K
        self.dim = dim
        if mu is None:
            mu = np.random.rand(K, dim)*20 - 10
        self.mu = torch.tensor(mu) # (K, D)
        if sigma is None :
            sigma = np.zeros((K, dim, dim))
            for i in range(K):
                sigma[i] = np.eye(dim)
        self.sigma = torch.tensor(sigma) # (K, D, D)
        # assert torch.allclose(self.sigma, torch.tensor(sigma))
        if pi is None:
            pi = np.ones(self.K)/self.K
        self.pi = torch.tensor(pi) # (K,)

    def log_prob(self, X):
        '''
        Compute the log-prob of each element in X
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of each element in X: log Sum_k pi_k * N( X_i | mu_k, sigma_k ))
        '''
        return self.gmm.log_prob(X)

    def prob(self, X):
        '''
        Computes the prob of each element in X
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of each element in X: Sum_k pi_k * N( X_i | mu_k, sigma_k )
        '''
        return torch.exp(self.log_prob(X))

    def sample(self, shape):
        return self.gmm.sample(shape)

class FlowerGMM(GMM):

    def __init__(self):
        CENTERS_SCALE = 6.
        MAJOR_VAR = 3.
        MINOR_VAR = 1.
        r_angles = np.linspace(0., 2 * np.pi, 7, endpoint=False)
        vs = np.array([[np.cos(ang), np.sin(ang)] for ang in r_angles])
        perp_vs = np.array([[-np.sin(ang), np.cos(ang)] for ang in r_angles])
        mus = np.array([[0., 0.],] + [CENTERS_SCALE * v for v in vs])
        R_matrices = np.stack([vs, perp_vs], axis=1)
        petal_sigmas = np.array([[MAJOR_VAR, 0.], [0., MINOR_VAR]])[np.newaxis,...].repeat(7, axis=0)
        sigmas = np.concatenate([
            np.eye(2)[np.newaxis,...], 
            R_matrices.transpose(0, 2, 1) @ petal_sigmas @ R_matrices])
        super().__init__(8, 2, mu=mus, sigma=sigmas)