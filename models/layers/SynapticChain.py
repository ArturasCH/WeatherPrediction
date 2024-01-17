import torch
from torch import nn
import snntorch as snn
import numpy as np
from tsl.nn.blocks.encoders.mlp import MLP
from typing_extensions import Literal

class SynapticChain(nn.Module):
    def __init__(self,
                 hidden_size = 32,
                 n_layers=3,
                 output_type: Literal["spike", "synaptic_current", "membrane_potential"] = "spike",
                 return_last=False,
                 temporal_reduction_factor = 2
                 ):
        super(SynapticChain, self).__init__()
        self.return_last = return_last
        self.output_type = output_type
        self.n_layers = n_layers
        self.temporal_reduction_factor = temporal_reduction_factor
        # synaptic = []
        # linear = []
        chain = []
        # expanded_dim = hidden_size * 2
        # self.expanded_projection = nn.Linear(in_features=hidden_size, out_features=expanded_dim)
        for i in range(self.n_layers):
            # feature_size = hidden_size * self._get_multiplier(i)
            # feature_size = hidden_size
            # synaptic.append(self._build_synaptic_layer(feature_size))
            # linear.append(nn.Linear(in_features=feature_size, out_features=feature_size))
            chain.append(self._build_synaptic_layer(hidden_size))
            chain.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            # chain.append(MLP(input_size=hidden_size, hidden_size=hidden_size*2, output_size=hidden_size, activation='prelu'))
        
        
        # self.synaptic = nn.ModuleList(synaptic)
        # self.linear = nn.ModuleList(linear)
        # feature_size = self._get_multiplier(n_layers - 1)
        # self.syn1 = self._build_synaptic_layer(hidden_size * self._get_multiplier(0))
        # self.l1 = nn.Linear(in_features=hidden_size * self._get_multiplier(0), out_features=hidden_size * self._get_multiplier(0))
        # self.syn2 = self._build_synaptic_layer(hidden_size * self._get_multiplier(1))
        # self.l2 = nn.Linear(in_features=hidden_size * self._get_multiplier(1), out_features=hidden_size * self._get_multiplier(1))
        self.chain = nn.Sequential(*chain)
        self.synaptic_out = self._build_synaptic_layer(hidden_size, output=True)
        self.linear_out = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.readout = MLP(input_size=hidden_size, hidden_size=hidden_size*2, output_size=hidden_size, activation='prelu')
        
    def forward(self, x):
        b, t, n, f = x.size()
        
        if not self.return_last:
            # temporal_threshold = t//self.temporal_reduction_factor
            # output = torch.zeros((b, temporal_threshold, n, f), device=x.device)
            output = torch.zeros((b, t, n, f), device=x.device)
        
        for timestep in range(t):
            timestep_data = x[:, timestep, :, :]
            spike = self.chain(timestep_data)
            
            spike, synaptic_current, membrane_potential = self.synaptic_out(spike)
            transformed = self.linear_out(spike)
            
            # if not self.return_last and timestep >= t - temporal_threshold:
                # output[:, timestep - temporal_threshold, :, :] = self._get_output(transformed, synaptic_current, membrane_potential)
            if not self.return_last:
                output[:, timestep, :, :] = self.readout(self._get_output(transformed, synaptic_current, membrane_potential))
                
        if self.return_last:
            output = self.readout(self._get_output(transformed, synaptic_current, membrane_potential))

        return output
    
    def _get_output(self, transformed, synaptic_current, membrane_potential):
        output_by_type = {
            "spike": transformed,
            "synaptic_current": synaptic_current,
            "membrane_potential": membrane_potential    
        }
        return output_by_type.get(self.output_type)
        #     for i, (synaptic, linear, (synaptic_current, membrane_potential)) in enumerate(zip(self.synaptic, self.linear, synaptic_inits)):
        #         spike, sc, mp = synaptic(timestep_data, synaptic_current, membrane_potential)
        #         transformed = linear(spike)
                
        #         timestep_data = torch.cat((timestep_data, transformed), dim=2)
        #         # synaptic_inits[i] = (sc, mp)
        #         synaptic_current, membrane_potential = sc, mp
                
            # spike = self.chain(timestep_data)
            # spike1 = self.syn1(timestep_data)
            # transformed1 = self.l1(spike)
            # spike2 = self.syn2(torch.cat((timestep_data, transformed1), dim=2))
            # transformed2 = self.l2(spike2)
            # spike, sc, mp = self.synaptic_out(torch.cat((transformed1, transformed2), dim=2))
            # transformed = self.linear_out(spike)
            
            
        
            # spikes[:, timestep, :, :] = transformed
            # synaptic_currents[:, timestep, :, :] = sc
            # membrane_potentials[:, timestep, :, :] = mp
            
            # spike, syn, membrane_potential = self.synaptic(x[:, timestep, :, :], syn, membrane_potential)
            # # spike, membrane_potential = self.leaky(x[:, timestep, :, :])
            # spikes[:, timestep, :] = spike
            # synaptic_pots[:, timestep, :] = syn
            # membrane_pots[:, timestep, :] = membrane_potential
            
            # self.spike = spike
            # self.membrane_potential = membrane_potential
        
        # spikes = spikes.reshape((b, t, n, -1))
        # synaptic_pots = synaptic_currents.reshape((b, t, n, -1))
        # membrane_pots = membrane_potentials.reshape((b, t, n, -1))
        
        # -------------------------------------
        # syn1, mem_p1 = self.syn1.init_synaptic()
        # syn2, mem_p2 = self.syn2.init_synaptic()
        # syn_out, mem_p_out = self.synaptic_out.init_synaptic()
        
        # for timestep in range(t):
        #     timestep_data = x[:, timestep, :, :]
        #     self._not_nan(timestep_data)
            
        #     spike1, syn1, mem_p1 = self.syn1(timestep_data, syn1, mem_p1)
        #     self._not_nan(spike1)
        #     self._not_nan(syn1)
        #     self._not_nan(mem_p1)
        #     transformed1 = self.l1(spike1)
        #     self._not_nan(transformed1)
            
        #     timestep_data = torch.cat((timestep_data, transformed1), dim=2)
        #     self._not_nan(timestep_data)
        #     spike2, syn2, mem_p2 = self.syn2(timestep_data, syn2, mem_p2)
        #     self._not_nan(spike2)
        #     self._not_nan(syn2)
        #     self._not_nan(mem_p2)
        #     transformed2 = self.l2(spike2)
        #     self._not_nan(transformed1)
            
        #     timestep_data = torch.cat((timestep_data, transformed2), dim=2)
        #     self._not_nan(timestep_data)
            
        #     spike_out, syn_out, mem_p_out = self.synaptic_out(timestep_data, syn_out, mem_p_out)
        #     self._not_nan(spike_out)
        #     self._not_nan(syn_out)
        #     self._not_nan(mem_p_out)
        #     transformed_out = self.linear_out(spike_out)
        #     self._not_nan(transformed_out)
            
        # return transformed_out, mem_p_out, syn_out
        # -------------------------------------
        
        # return transformed, mp, sc
        # if self.return_last:
            # return spikes[:, -1, :, :], membrane_potentials[:, -1, :, :], synaptic_currents[:, -1, :, :]
            # return spikes[:, -1, :, :]
        # else:
        #     return spikes, membrane_potentials, synaptic_currents
            # return spikes
            
    def _build_synaptic_layer(self, in_size, output=False):
        return snn.Synaptic(
            # alpha=torch.Tensor(in_size),
            # beta=torch.Tensor(in_size),
            alpha=0.9,
            beta=0.8,
            learn_alpha=True,
            learn_beta=True,
            learn_threshold=True,
            init_hidden=True,
            output=output
            )
    
    def _get_multiplier(self, i):
        return max((i * 2), 1)
    
    def _not_nan(self, x):
        x = x.cpu()
        nan_mask = torch.isnan(x)
        np.savetxt('./nan_tensor.txt', x.detach().numpy().flatten())
        if nan_mask.any():
            nan_indices = nan_mask.nonzero()
            raise RuntimeError(f"found nan in input {x} at indices: {nan_indices}, where {x[nan_mask.nonzero()[:, 0].unique(sorted=True)]}")