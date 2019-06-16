import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class Policy(nn.Module):
    """a linear policy network

    Parameters
    ----------
    dim_input : type
        Description of parameter `dim_input`.
    dim_output : type
        Description of parameter `dim_output`.

    Attributes
    ----------
    affine : type
        Description of attribute `affine`.

    """

    def __init__(self, dim_input, dim_output):
        super(Policy, self).__init__()
        self.affine = nn.Linear(dim_input, dim_output)

    def forward(self, x):
        output = self.affine(x)
        action_probs = F.softmax(output, dim=1)
        return action_probs

    def choose_action(self, s_t):
        prob = self.forward(s_t)
        m = Categorical(prob)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t
