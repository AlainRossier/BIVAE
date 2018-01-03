import torch
import numpy as np
from random import randint
from torch.autograd import Variable

def sample(mu, var, nb_samples=500):
    """
    :param mu: torch.Tensor (features)
    :param var: torch.Tensor (features) (note: zero covariance)
    :return: torch.Tensor (nb_samples, features)
    """
    K = mu.size(0)
    out = []
    for i in range(nb_samples):
        label = randint(0, K-1)
        j = 2 * label
        out.append([
            torch.normal(mu, var.sqrt()).view(1, -1).narrow(1, j, 2).squeeze(0), label%2
        ])
    return out

def initialize(data, K, var=1):
  """
  :param data: design matrix (examples, features)
  :param K: number of gaussians
  :param var: initial variance
  """
  # choose k points from data to initialize means
  M = data.size(0) # nb. of training examples
  idxs = Variable(torch.from_numpy(np.random.choice(M, K, replace=False)))
  mu = data[idxs]

  # fixed variance
  N = data.size(1) # nb. of features
  var = Variable(torch.Tensor(K, N).fill_(var))

  # equal priors
  pi = Variable(torch.Tensor(K).fill_(1)) / K

  return mu, var, pi

def get_k_likelihoods(X, Y, mu, var):
    """
    Compute the densities of each data point under the parameterised gaussians.

    :param X: design matrix (examples=N, features=d)
    :param Y: variance of the design matrix (N, d)
    :param mu: the component means (K, d)
    :param var: the component variances (K, d)
    :return: relative likelihoods (K, N)
    """

    if var.data.eq(0).any():
        raise Exception('variances must be nonzero')

    K = mu.size(0)
    N = X.size(0)
    X = X.unsqueeze(0).repeat(K, 1, 1)
    Y = Y.unsqueeze(0).repeat(K, 1, 1)
    var = var.unsqueeze(1).repeat(1, N, 1)
    #X, Y, var are now (K, N, d)
    
    # get the diag of the inverse covar. matrix
    covar_inv = 1. / (var+Y) # (K, N, d)

    # compute the coefficient
    det = (2 * np.pi * (var+Y)).prod(dim=2) # (K, N)
    coeff = 1. / det.sqrt() # (K, N)

    # calculate the exponent
    a = (X - mu.unsqueeze(1)) # (K, N, d)
    exponent = (a ** 2) * covar_inv
    exponent = torch.sum(exponent, dim=2)
    exponent = -0.5 * exponent # (K, N)

    # compute probability density
    return coeff * exponent.exp() # (K, N)

def get_posteriors(P, eps=1e-6):
    """
    :param P: the relative likelihood of each data point under each gaussian (K, examples)
    :return: p(z|x): (K, examples)
    """
    P_sum = torch.sum(P, dim=0, keepdim=True)
    return (P / (P_sum+eps))

def get_parameters(X, gamma, eps=1e-6):
  """
  :param X: design matrix (examples=N, features=d)
  :param gamma: the posterior probabilities p(z|x) (K, N)
  :returns mu, var, pi: (K, d) , (K, d) , (K)
  """

  # compute `N_k` the proxy "number of points" assigned to each distribution.
  K = gamma.size(0)
  N_k = torch.sum(gamma, dim=1) + eps # (K)
  N_k = N_k.view(K, 1, 1)

  # tile X on the `K` dimension
  X = X.unsqueeze(0).repeat(K, 1, 1)

  # get the means by taking the weighted combination of points
  mu = gamma.unsqueeze(1) @ X # (K, 1, features)
  mu = mu / N_k

  # compute the diagonal covar. matrix, weighting contributions from each point
  A = X - mu
  var = gamma.unsqueeze(1) @ (A ** 2) # (K, 1, features)
  var = var / N_k

  # recompute the mixing probabilities
  m = X.size(1) # nb. of training examples
  pi = N_k / N_k.sum()

  return mu.squeeze(1), var.squeeze(1), pi.view(-1)



def log_likelihood(P, pi, eps=1e-6):
    """
    Get the log-likelihood of the data points under the given distribution.
    :param P: likelihoods / densities under the distributions. (K, examples)
    :param pi: priors (K)
    """
    
    # get weight probability of each point under each k
    sum_over_k = torch.sum(pi.unsqueeze(1) * P, dim=0)
    
    # take log probability over each example `m`
    sum_over_m = torch.sum(torch.log(sum_over_k + eps))
    
    # divide by number of training examples
    return - sum_over_m / P.size(1)


def EM_algo(X, Y, k, nb_iter = 1000, thres=1e-4):
    """
    :param X : design matrix Nxd, param Y : its variance
    :k : number of gaussians
    """
    d = X.size()[1]
    mu, var, pi = initialize(X, k, var=1)
    prev_cost = float('inf')
    for i in range(nb_iter):
        P = get_k_likelihoods(X, Y, mu, var)
        gamma = get_posteriors(P)
        cost = log_likelihood(P, pi)
        diff = prev_cost - cost
        if torch.abs(diff).data[0] < thres:
            break
        prev_cost = cost
        mu, var, pi = get_parameters(X, gamma)
    return mu, var, pi

    
if __name__ == "__main__":
    mu0 = torch.Tensor([[3,5], [-2, 3], [4, -2]])
    var0 = torch.Tensor([[1,1], [1,1], [1,1]])
    X = Variable(sample(mu0, var0, 500))
    Y = Variable(torch.ones(500, 2))
    thefile = open("GMM.data", 'w')
    for j in range(500):
        thefile.write("%.2f %.2f \n" %(X.data[j,0], X.data[j, 1]))
    thefile.close()
    mu, var, pi = EM_algo(X, Y, 3)
    print("mu : ", mu)
    print("var : ", var)
    print("pi : ", pi)
