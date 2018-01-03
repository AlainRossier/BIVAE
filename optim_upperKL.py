import numpy as np
import torch
from torch.autograd import Variable
from datetime import datetime

#the 4 arguments are 1d tensor, defines 2 multivariate gaussian distribution
#computes KL(N(mu1, sigma1) || N(mu2, sigma2))
def KL_div_gaussian(mu1, logvar1, mu2, logvar2):
    KL_element_1 = logvar2.sub(logvar1).sub(1)
    KL_element_2 = logvar1.exp().div(logvar2.exp())
    KL_element_3 = mu1.sub(mu2).pow(2).div(logvar2.exp())
    return 0.5*torch.sum(KL_element_1.add(KL_element_2).add(KL_element_3))

#form the 2x2 matrix as in our theory.
def KL_matrix(mu1, mu2, logvar, batch_size):
    n = mu1.size()[1]
    l = Variable(torch.Tensor(2, 2))
    l[0,0] = KL_div_gaussian(mu1, logvar, Variable(torch.ones(batch_size, n)),
                             Variable(torch.zeros(batch_size, n)))
    l[0,1] = KL_div_gaussian(mu2, logvar, Variable(torch.ones(batch_size, n)),
                             Variable(torch.zeros(batch_size, n)))
    l[1,0] = KL_div_gaussian(mu1, logvar, Variable(torch.ones(batch_size, n).mul(-1)),
                             Variable(torch.zeros(batch_size, n)))
    l[1,1] = KL_div_gaussian(mu2, logvar, Variable(torch.ones(batch_size, n).mul(-1)),
                             Variable(torch.zeros(batch_size, n)))
    return l.div_(batch_size)

#mu is a 2xdim_latent vector
#sigma is a 2xdim_latent vector
def KL_matrix_new(mu, var):
    n = mu.size()[1]
    l = Variable(torch.Tensor(2, 2))
    l[0, 0] = KL_div_gaussian(mu.narrow(0, 0, 1), var.narrow(0, 0, 1).log(),
                              Variable(torch.ones(n)), Variable(torch.zeros(n)))
    l[0, 1] = KL_div_gaussian(mu.narrow(0, 0, 1), var.narrow(0, 0, 1).log(),
                              Variable(torch.ones(n).mul(-1)), Variable(torch.zeros(n)))
    l[1, 0] = KL_div_gaussian(mu.narrow(0, 1, 1), var.narrow(0, 1, 1).log(),
                              Variable(torch.ones(n)), Variable(torch.zeros(n)))
    l[1, 1] = KL_div_gaussian(mu.narrow(0, 1, 1), var.narrow(0, 1, 1).log(),
                              Variable(torch.ones(n).mul(-1)), Variable(torch.zeros(n)))

    return l

#weights_f is an m-dim vector that contains weights for GMM f, similar for g (n-dim), sum to 1
#indiv_KL is a mxn matrix, the ij entry is the KL(f_i, g_j)
def optim_parameters(indiv_KL, weights_f, weights_g):
    #initialization
    m = weights_f.size()[0]
    n = weights_g.size()[0]
    weights_f_matrix = weights_f.resize(m, 1).expand(m ,n)
    weights_g_matrix = weights_g.resize(n ,1).expand(n ,m)
    phi = torch.ones(m, n)
    psi = torch.ones(n, m)
    phi.mul_(weights_f_matrix.data).div_(n)
    psi.mul_(weights_g_matrix.data).div_(m)
    phi = Variable(phi)
    psi = Variable(psi)

    #iteration
    count = 0
    while(count < 10): #find a better way to iterate
        #optimize phi
        energy_phi = psi.t().mul(indiv_KL.mul(-1).exp())
        partition_phi = torch.sum(energy_phi, 1).resize(m ,1).expand(m ,n)
        phi = energy_phi.mul(weights_f_matrix).div(partition_phi)
        #optimize psi
        energy_psi = phi.t()
        partition_psi = torch.sum(energy_psi, 1).resize(n, 1).expand(n ,m)
        psi = energy_psi.mul(weights_g_matrix).div(partition_psi)
        count += 1
    return phi, psi

#compute the KL divergence of two discrete distribution of size mxn
def KL_discrete(p, q):
    p = p.add(pow(10, -4))
    q = q.add(pow(10, -4))
    return torch.sum(p.mul(p.log().sub(q.log())))
    
def upper_bound_KL(indiv_KL, weights_f, weights_g):
    phi, psi = optim_parameters(indiv_KL, weights_f, weights_g)
    return KL_discrete(phi, psi.t()) + torch.sum(indiv_KL.mul(phi))



############## Example of a program running #################

if __name__ == "__main__":

    startTime = datetime.now()
    batch_size = 2
    weights_f = Variable(torch.Tensor([0.5, 0.5]))
    weights_g = Variable(torch.Tensor([0.5, 0.5]))
    mu1 = Variable(torch.Tensor([[1, -1, 0], [0, 0, 0]]).resize_(batch_size, 3), requires_grad=True)
    mu2 = Variable(torch.Tensor([[-1, 0, 1], [0, 0, 0]]).resize_(batch_size, 3), requires_grad=True)
    logvar = Variable(torch.Tensor([[1, 1, 1], [1, 1, 1]]).resize_(batch_size, 3), requires_grad=True)
    m = KL_matrix(mu1, mu2, logvar, batch_size)
    KL_upper = upper_bound_KL(m, weights_f, weights_g)
    KL_upper.backward()
    print(mu1.grad)
    print(mu2.grad)
    
    #phi, psi = optim_parameters(indiv_KL, weights_f, weights_g)
    #print(phi, psi)
    #print(KL_discrete(phi, psi.t()))
    #print(upper_bound_KL(indiv_KL, weights_f, weights_g))
    
    print("Time taken : ", (datetime.now() - startTime).total_seconds())
    

    
    #print(KL_div_gaussian(mu1, logvar1, mu2, logvar2))
