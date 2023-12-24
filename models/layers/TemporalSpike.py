import torch
from torch import nn
import snntorch as snn
from snntorch import utils
from snntorch import surrogate

class TemporalSpike(nn.Module):
    def __init__(self, hidden_size = 32, beta=0.9, return_last=False, spike_grad=surrogate.atan(alpha=2.0),
                 thresh=1) -> None:
        super(TemporalSpike, self).__init__()
        self.rlif = snn.RLeaky(beta=beta, linear_features=hidden_size)
        # self.leaky = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=thresh, output=True)
        self.return_last = return_last
        
    def forward(self, x):
        """
        Args:
            x (tensor): (b,t,n,f)
        """
        spike, membrane_potential = self.rlif.init_rleaky()
        b, t, n, f = x.size()
        # utils.reset(self.leaky)
        spikes = torch.zeros(x.size()).cuda()
        membrane_pots = torch.zeros(x.size()).cuda()
        for timestep in range(t):
            spike, membrane_potential = self.rlif(x[:, timestep, :, :], spike, membrane_potential)
            # spike, membrane_potential = self.leaky(x[:, timestep, :, :])
            spikes[:, timestep, :, :] = spike
            membrane_pots[:, timestep, :, :] = membrane_potential
            
            # self.spike = spike
            # self.membrane_potential = membrane_potential
            
        if self.return_last:
            return spikes[:, -1, :, :], membrane_pots[:, -1, :, :]
            # return spikes[:, -1, :, :]
        else:
            return spikes, membrane_pots
            # return spikes