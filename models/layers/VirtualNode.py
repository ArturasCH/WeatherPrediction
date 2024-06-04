import torch
import torch.nn as nn

class VirtualNode(nn.Module):
    def __init__(self, n_nodes: int, n_channels: int):
        super(VirtualNode, self).__init__()
        self.virtual_node = torch.zeros((1, n_channels)).unsqueeze(0).unsqueeze(0)
        self.v_node_idx = n_nodes
        existing_node = torch.arange(0, self.v_node_idx)
        virtual_node = torch.full((self.v_node_idx,1), self.v_node_idx).squeeze()
        messages_to_virtual_node = torch.stack([existing_node, virtual_node])
        messages_from_virtual_node = torch.stack([virtual_node, existing_node])
        self.virtual_node_edge_index = torch.cat([messages_to_virtual_node, messages_from_virtual_node], dim=-1)
        
        self.edge_index_with_virtual_node = None
        
        
    def forward(self, x, edge_index):
        if x.size(-2) > self.v_node_idx:
            x = x[:,:,:self.v_node_idx]
            
        x_with_virtual_node = torch.cat([x, self.virtual_node.expand(x.size(0), x.size(1), -1, -1).to(x.device)], dim=-2)
        
        return x_with_virtual_node, self.get_edge_index_with_virtual_node(edge_index)
        
    def get_edge_index_with_virtual_node(self, edge_index):
        if self.edge_index_with_virtual_node == None:
            if self.v_node_idx in edge_index[0]:
                return edge_index
            edge_index_with_virtual_node = torch.cat([edge_index, self.virtual_node_edge_index.to(edge_index.device)], dim=-1)
            self.edge_index_with_virtual_node = edge_index_with_virtual_node
        return self.edge_index_with_virtual_node