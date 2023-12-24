import torch
from torch import nn
import snntorch as snn

class SynapticSpike(nn.Module):
    def __init__(self, n_nodes, hidden_size = 32, return_last=False, timesteps=112):
        super(SynapticSpike, self).__init__()
        self.return_last = return_last
        
        # alpha = torch.nn.Parameter(data=torch.Tensor(n_nodes * hidden_size), requires_grad=True)
        # beta = torch.nn.Parameter(data=torch.Tensor(n_nodes * hidden_size), requires_grad=True)
        # self.synaptic = snn.Synaptic(alpha=alpha, beta=beta)
        
        # self.rsynaptic = snn.RSynaptic(
        #     linear_features=hidden_size,
        #     alpha=torch.Tensor(hidden_size),
        #     beta=torch.Tensor(hidden_size),
        #     learn_alpha=True,
        #     learn_beta=True,
        #     learn_recurrent=True,
        #     learn_threshold=True)
        self.rsynaptic_conv = snn.RSynaptic(
            conv2d_channels=timesteps,
            kernel_size=3,
            alpha=torch.Tensor(hidden_size),
            beta=torch.Tensor(hidden_size),
            learn_alpha=True,
            learn_beta=True,
            learn_recurrent=True,
            learn_threshold=True)
        
        # self.linear = torch.nn.Linear()
        
    # def forward(self, x):
    #     b, t, n, f = x.size()
    #     # syn, membrane_potential = self.synaptic.init_synaptic()
        
    #     flat_x = x.flatten(start_dim=2)
    #     spikes = torch.zeros(flat_x.size()).cuda()
    #     synaptic_pots = torch.zeros(flat_x.size()).cuda()
    #     membrane_pots = torch.zeros(flat_x.size()).cuda()
        
        
    #     for timestep in range(t):
    #         spike, syn, membrane_potential = self.synaptic(flat_x[:, timestep, :], syn, membrane_potential)
    #         # spike, membrane_potential = self.leaky(x[:, timestep, :, :])
    #         spikes[:, timestep, :] = spike
    #         synaptic_pots[:, timestep, :] = syn
    #         membrane_pots[:, timestep, :] = membrane_potential
            
    #         # self.spike = spike
    #         # self.membrane_potential = membrane_potential
        
    #     spikes = spikes.reshape((b, t, n, -1))
    #     synaptic_pots = synaptic_pots.reshape((b, t, n, -1))
    #     membrane_pots = membrane_pots.reshape((b, t, n, -1))
        
    #     if self.return_last:
    #         return spikes[:, -1, :, :], membrane_pots[:, -1, :, :], synaptic_pots[:, -1, :, :]
    #         # return spikes[:, -1, :, :]
    #     else:
    #         return spikes, membrane_pots, synaptic_pots
    #         # return spikes
    
    
    def forward(self, x):
        b, t, n, f = x.size()
        # syn, membrane_potential = self.synaptic.init_synaptic()
        # spike, syn, membrane_potential = self.rsynaptic.init_rsynaptic()
            
        spike, syn, membrane_potential = self.rsynaptic_conv.init_rsynaptic()
        
        # flat_x = x.flatten(start_dim=2)
        # spikes = torch.zeros(x.size()).cuda()
        # synaptic_pots = torch.zeros(x.size()).cuda()
        # membrane_pots = torch.zeros(x.size()).cuda()
        
        
        # for timestep in range(t):
        #     spike, syn, membrane_potential = self.rsynaptic_conv(x[:, timestep, :, :], spike, syn, membrane_potential)
        #     # spike, membrane_potential = self.leaky(x[:, timestep, :, :])
        #     spikes[:, timestep, :] = spike
        #     synaptic_pots[:, timestep, :] = syn
        #     membrane_pots[:, timestep, :] = membrane_potential
            
            # self.spike = spike
            # self.membrane_potential = membrane_potential
        
        # spikes = spikes.reshape((b, t, n, -1))
        # synaptic_pots = synaptic_pots.reshape((b, t, n, -1))
        # membrane_pots = membrane_pots.reshape((b, t, n, -1))
        spikes, synaptic_pots, membrane_pots = self.rsynaptic_conv(x, spike, syn, membrane_potential)
        
        if self.return_last:
            return spikes[:, -1, :, :], membrane_pots[:, -1, :, :], synaptic_pots[:, -1, :, :]
            # return spikes[:, -1, :, :]
        else:
            return spikes, membrane_pots, synaptic_pots
            # return spikes