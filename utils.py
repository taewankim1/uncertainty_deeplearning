import math
import torch
import numpy as np
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
def gaussian_probability(mu, sigma, data):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
#     target = target.unsqueeze(1).expand_as(sigma)
    ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / sigma)**2) / sigma
#     print(ret.shape)
#     return ret
    return torch.prod(ret, 2)


def mdn_loss(mu, sigma, data):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = gaussian_probability(sigma, mu, data)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def log_gaussian_loss(output, sigma, target, dim):
    exponent = 0.5*(target - output)**2/sigma**2
#     log_coeff = -dim*torch.log(sigma) - 0.5*dim*np.log(2*np.pi)
    log_coeff = torch.log(sigma)
    
    return (log_coeff + exponent).sum()
