import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from typing import Callable

class MemoryStore(torch.nn.Module):
    def __init__(self, update_fn: Callable, readout: Callable, memory_size=(10, 66), key_sequence_size=10):
        super(MemoryStore, self).__init__()
        # self.mem_slots = n_memory_slots
        # self.n_nodes = n_nodes
        # self.n_features = n_features
        self.update_fn = update_fn
        self.readout = readout
        
        self.register_buffer('memory', torch.zeros(memory_size))
        self.register_buffer('update_count', torch.zeros(1))
        self.register_buffer('update_counter', torch.zeros(memory_size[0]))
        
        
        self.get_memory_index = torch.nn.Sequential(
            torch.nn.Linear(key_sequence_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, memory_size[0]),
        )
        
    def forward(self, indicator, x):
        indices = self._get_memory_index(indicator)
        # return torch.einsum('bm, mnf -> bnf', probs, self.memory)
        return self.readout(self.memory.detach()[indices], x)
    
    def update_state(self, indicator, data):
        # data - b t n f
        # with torch.no_grad():
        indices = self._get_memory_index(indicator)
        # # Ex(t+1) = Ex(t) + (1/(t+1)) * (x - Ex(t)) 
        
        # local_update_counter = torch.zeros_like(self.update_counter)
        # local_update_counter[indices] += indices.bincount()[indices]
        
        # update_data = data.mean(dim=1)
        # # print(update_data.size(), indices, local_update_counter[indices])
        # self.update_counter[indices] += 1
        # sum_store = torch.zeros_like(self.memory)
        # sum_store = sum_store.index_add(0, indices, update_data)
        # mean_update = (sum_store[indices] / local_update_counter[indices].unsqueeze(-1).unsqueeze(-1).expand(-1, update_data.size(1), update_data.size(2)))
        # mean_calc_res = self.memory[indices] + (mean_update - self.memory[indices]) * (1 / self.update_counter[indices].unsqueeze(-1).unsqueeze(-1).expand(-1, update_data.size(1), update_data.size(2)))
        # iterative mean update
        
        # mean_calc_res = self.update_fn(self.memory, data, indices, self.update_counter)
        # self.memory[indices] = self.update_fn(self.memory, data, indices, self.update_counter)
        self.memory = self.memory.clone()
        if self.training:
            update = self.update_fn(self.memory, data, indices, self.update_counter)
            self.memory[indices] = update
            self.update_counter[indices] += 1
            
                
    def _get_memory_index(self, indicator):
        probs = torch.softmax(self.get_memory_index(indicator), dim=1) #expecting batched input
        indices = torch.argmax(probs, dim=-1)
        
        return indices.to(torch.long)
    
    
class MeanUpdate(nn.Module):
    def __init__(self) -> None:
        super(MeanUpdate, self).__init__()
    def forward(self, memory, data, indices, update_counter):
        local_update_counter = torch.zeros_like(update_counter)
        local_update_counter[indices] += indices.bincount()[indices]
        update_counter[indices] += 1
        update_data = data.mean(dim=1)
        
        sum_store = torch.zeros_like(memory)
        sum_store = sum_store.index_add(0, indices, update_data)
        mean_update = (sum_store[indices] / local_update_counter[indices].unsqueeze(-1).unsqueeze(-1).expand(-1, update_data.size(1), update_data.size(2)))
        return memory[indices] + (mean_update - memory[indices]) * (1 / update_counter[indices].unsqueeze(-1).unsqueeze(-1).expand(-1, update_data.size(1), update_data.size(2)))
    
class IdentityReadout(nn.Module):
    def __init__(self) -> None:
        super(IdentityReadout, self).__init__()
    def forward(self, memory, x):
        return memory

from torch.utils.checkpoint import checkpoint

# class GRUReadout(nn.Module):
#     def __init__(self, in_features, hidden_size, n_layers=1, bidirectional=False):
#         super(GRUReadout, self).__init__()
#         self.gru = torch.nn.GRU(in_features, hidden_size, n_layers, bidirectional=bidirectional)
#         self.n_layers = n_layers
#         self.flatten_x = Rearrange('b t n f -> b n t f')
#         self.reconstruct_x = Rearrange('b n t f -> b t n f')
        
#         self.expansion_rate = n_layers * 2 if bidirectional else n_layers
#         # self.readout = nn.Linear(in_features, hidden_size)
        
#         # self.flatten_memory = Rearrange('b n f -> b (n f)')
#         # self.reconstruct_memory = Rearrange('b (n f) -> b n f', f = in_features)
        
#     def forward(self, memory):
#         """
#         Args:
#             memory (Tensor): b n f
#         """
#         # flattened_input = self.flatten_x(x)
#         # # flattened_memory = self.flatten_memory(memory)
#         # batches = []
#         # for i in range(x.size(0)):
#         #     inp = flattened_input[i]
#         #     m = memory[i]
#         # res, hidden_state = checkpoint(self.gru, inp, m.unsqueeze(0).expand(self.expansion_rate, -1, -1), use_reentrant=True)
#         # print(f'res: {res.size()}')
#         # batches.append(res)
#         # stacked = torch.stack(batches, dim=0)
#         # print(stacked.size())
#         # return self.reconstruct_x(stacked)
#         res, hidden = self.gru(memory)
#         return res

# class GRUMemoryUpdate(nn.Module):
#     def __init__(self, in_features, hidden_size, n_layers=1, bidirectional=False):
#         super(GRUMemoryUpdate, self).__init__()
#         self.gru = torch.nn.GRU(in_features, hidden_size,n_layers, bidirectional=bidirectional)
#         self.n_layers = n_layers
#         self.expansion_rate = n_layers * 2 if bidirectional else n_layers
#         self.flatten_x = Rearrange('b t n f -> b n t f')
#         self.reconstruct_x = Rearrange('b n t f -> b t n f')
        
#         self.flatten_memory = Rearrange('b n f -> b (n f)')
#         self.reconstruct_memory = Rearrange('b (n f) -> b n f', f = in_features)
        
#         self.readout = nn.Linear(in_features, hidden_size)
        
        
#     def forward(self, memory, data, indices, update_counter):
#         b = data.size(0)
#         # flattened_input = self.flatten_x(x)
#         # flattened_memory = self.flatten_memory(memory[indices])
        
#         # res, hidden_state = checkpoint(self.gru, flattened_input, flattened_memory.unsqueeze(0).expand(self.n_layers, -1, -1), use_reentrant=True)
        
#         # return self.reconstruct_x(res[:, -1]) #return last state as new memory
#         batches = []
#         for i in range(b):
#             inp = data[i]
#             m = memory[indices][i]
#             # runs time as batches and nodes as sequence, but is a sort of aggregation
#             res, hidden = self.gru(inp, m.unsqueeze(0).expand(self.expansion_rate, -1, -1))
#             batches.append(res)
#         stacked = torch.stack(batches, dim=0)
#         return self.readout(stacked.mean(dim=1))


from torch.utils.checkpoint import checkpoint

class GRUReadout(nn.Module):
    def __init__(self, in_features, hidden_size, n_layers=1, bidirectional=False):
        super(GRUReadout, self).__init__()
        self.gru = torch.nn.GRU(in_features, hidden_size, n_layers, bidirectional=bidirectional)
        self.n_layers = n_layers
        self.flatten_x = Rearrange('b t n f -> b n t f')
        self.reconstruct_x = Rearrange('b n t f -> b t n f')
        
        self.expansion_rate = n_layers * 2 if bidirectional else n_layers
        
        # self.flatten_memory = Rearrange('b n f -> b (n f)')
        # self.reconstruct_memory = Rearrange('b (n f) -> b n f', f = in_features)
        
    def forward(self, memory, x):
        """
        Args:
            memory (Tensor): b n f
        """
        flattened_input = self.flatten_x(x)
        # # flattened_memory = self.flatten_memory(memory)
        batches = []
        for i in range(x.size(0)):
            inp = flattened_input[i]
            m = memory[i]
            m = m.unsqueeze(0).unsqueeze(0).expand(self.expansion_rate, inp.size(1), -1).contiguous()
            res, hidden_state = checkpoint(self.gru, inp, m, use_reentrant=True)
            batches.append(res)
        stacked = torch.stack(batches, dim=0)
        return self.reconstruct_x(stacked)
        # return self.gru(memory)[0]

class GRUMemoryUpdate(nn.Module):
    def __init__(self, in_features, hidden_size, n_layers=1, bidirectional=False):
        super(GRUMemoryUpdate, self).__init__()
        self.gru = torch.nn.GRU(in_features, hidden_size,n_layers, bidirectional=bidirectional)
        self.n_layers = n_layers
        self.expansion_rate = n_layers * 2 if bidirectional else n_layers
        self.flatten_x = Rearrange('b t n f -> b n t f')
        self.reconstruct_x = Rearrange('b n t f -> b t n f')
        
        self.flatten_memory = Rearrange('b n f -> b (n f)')
        self.reconstruct_memory = Rearrange('b (n f) -> b n f', f = in_features)
        
        
    def forward(self, memory, data, indices, update_counter):
        b = data.size(0)
        flattened_input = self.flatten_x(data)
        # flattened_memory = self.flatten_memory(memory[indices])
        
        # res, hidden_state = checkpoint(self.gru, flattened_input, flattened_memory.unsqueeze(0).expand(self.n_layers, -1, -1), use_reentrant=True)
        
        # return self.reconstruct_x(res[:, -1]) #return last state as new memory
        batches = []
        for i in range(b):
            inp = flattened_input[i]
            m = memory[indices][i]
            m = m.unsqueeze(0).unsqueeze(0).expand(self.expansion_rate, inp.size(1), -1).contiguous()
            # runs time as batches and nodes as sequence, but is a sort of aggregation
            res, hidden = checkpoint(self.gru, inp, m, use_reentrant=True)
            batches.append(hidden)
        stacked = torch.stack(batches, dim=0).squeeze()
        return stacked.mean(dim=1)
