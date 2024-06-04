import h3
import torch
import torch.nn as nn
import numpy as np
import einops
from torch_geometric.data import Data
from models.layers.MLPGraphEncoder import MLPGraphEncoder
from torch_geometric.nn.models import MLP
from tsl.nn.blocks import ResidualMLP


class FineToCoarseEncoder(nn.Module):
    def __init__(self,
                lat_lons,
                resolution=1,
                input_dim: int = 78,
                output_dim: int = 256,
                output_edge_dim: int = 256,
                hidden_dim_processor_node=256,
                hidden_dim_processor_edge=256,
                hidden_layers_processor_node=2,
                hidden_layers_processor_edge=2,
                 ):
        super(FineToCoarseEncoder, self).__init__()
        self.base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
        self.base_h3_map = {h_i: i for i, h_i in enumerate(self.base_h3_grid)}
        h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        h3_mapping = {}
        h_index = len(self.base_h3_grid)

        for h in self.base_h3_grid:
            if h not in h3_mapping:
                h_index -= 1
                h3_mapping[h] = h_index

        h3_distances = []
        for idx, h3_point in enumerate(h3_grid):
            lat_lon = lat_lons[idx]
            distance = h3.point_dist(lat_lon, h3.h3_to_geo(h3_point), unit="rads")
            h3_distances.append([np.sin(distance), np.cos(distance)])
        h3_distances = torch.tensor(h3_distances, dtype=torch.float)

        edge_sources = []
        edge_targets = []
        for node_index, lat_node in enumerate(h3_grid):
            edge_sources.append(node_index)
            edge_targets.append(h3_mapping[lat_node])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        self.graph = Data(edge_index=edge_index, edge_attr=h3_distances)
        self.h3_nodes = torch.nn.Parameter(
            torch.zeros((h3.num_hexagons(resolution), input_dim), dtype=torch.float)
        )
        
        self.latent_graph = self.create_latent_graph()
        self.h3_nodes = torch.nn.Parameter(
            torch.zeros((h3.num_hexagons(resolution), input_dim), dtype=torch.float)
        )
        
        self.graph_encoder = MLPGraphEncoder(
            in_channels=input_dim,
            hidden_channels=hidden_dim_processor_node,
            out_channels=output_dim,
            edge_in_channels=self.graph.edge_attr.size(-1),
            edge_hidden_channels=hidden_dim_processor_edge,
            edge_out_channels=output_edge_dim,
            n_node_layers=hidden_layers_processor_node,
            n_edge_layers=hidden_layers_processor_edge
        )
        
        self.latent_edge_encoder = ResidualMLP(
                input_size=2,
                hidden_size=hidden_dim_processor_edge,
                output_size=output_edge_dim,
                n_layers=hidden_layers_processor_edge,
                activation='leaky_relu',
                dropout=0.3,
                parametrized_skip=True,
            )
        
        
    def forward(self, x):
        b,t,n,f = x.size()
        
        # roll batches together with timesteps to do everything in a single pass
        batch_size =  b * t
        x = einops.rearrange(x, "b t n f -> (b t) n f")
        
        latent_nodes = einops.repeat(self.h3_nodes, "n f -> b n f", b=batch_size)
        encoded_coarse_graph, _ = self.graph_encoder((x, latent_nodes.to(x.device)), self.graph.edge_index.to(x.device), self.graph.edge_attr.to(x.device))
        encoded_attrs = self.latent_edge_encoder(self.latent_graph.edge_attr.to(x.device))
        
        return einops.rearrange(encoded_coarse_graph, '(b t) n f -> b t n f', b=b, t=t), self.latent_graph.edge_index.to(x.device), encoded_attrs
        
        
    def create_latent_graph(self):
        edge_sources = []
        edge_targets = []
        edge_attrs = []
        for h3_index in self.base_h3_grid:
            h_points = h3.k_ring(h3_index, 1)
            for h in h_points:  # Already includes itself
                distance = h3.point_dist(h3.h3_to_geo(h3_index), h3.h3_to_geo(h), unit="rads")
                edge_attrs.append([np.sin(distance), np.cos(distance)])
                edge_sources.append(self.base_h3_map[h3_index])
                edge_targets.append(self.base_h3_map[h])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
        # Use heterogeneous graph as input and output dims are not same for the encoder
        # Because uniform grid now, don't need edge attributes as they are all the same
        return Data(edge_index=edge_index, edge_attr=edge_attrs)

class CoarseGraphDecoder(nn.Module):
    def __init__(self,
                lat_lons,
                resolution = 1,
                input_dim: int = 256,
                output_dim: int = 256,
                output_edge_dim: int = 256,
                hidden_dim_processor_node=256,
                hidden_dim_processor_edge=256,
                hidden_layers_processor_node=2,
                hidden_layers_processor_edge=2,
                ):
        super(CoarseGraphDecoder, self).__init__()
        base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
        num_h3 = len(base_h3_grid)
        h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        h3_to_index = {}
        h_index = len(base_h3_grid)
        for h in base_h3_grid:
            if h not in h3_to_index:
                h_index -= 1
                h3_to_index[h] = h_index
        h3_mapping = {}
        for h, value in enumerate(h3_grid):
            h3_mapping[h + num_h3] = value

        # Build the default graph
        # Extra starting ones for appending to inputs, could 'learn' good starting points
        self.latlon_nodes = torch.zeros((len(lat_lons), input_dim), dtype=torch.float)
        # Get connections between lat nodes and h3 nodes TODO Paper makes it seem like the 3
        #  closest iso points map to the lat/lon point Do kring 1 around current h3 cell,
        #  and calculate distance between all those points and the lat/lon one, choosing the
        #  nearest N (3) For a bit simpler, just include them all with their distances
        edge_sources = []
        edge_targets = []
        h3_to_lat_distances = []
        for node_index, h_node in enumerate(h3_grid):
            # Get h3 index
            h_points = h3.k_ring(h3_mapping[node_index + num_h3], 1)
            for h in h_points:
                distance = h3.point_dist(lat_lons[node_index], h3.h3_to_geo(h), unit="rads")
                h3_to_lat_distances.append([np.sin(distance), np.cos(distance)])
                edge_sources.append(h3_to_index[h])
                # edge_targets.append(node_index + num_h3)
                edge_targets.append(node_index)
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        h3_to_lat_distances = torch.tensor(h3_to_lat_distances, dtype=torch.float)

        # Use normal graph as its a bit simpler
        self.graph = Data(edge_index=edge_index, edge_attr=h3_to_lat_distances)
        
        self.coarse_to_fine = MLPGraphEncoder(
            in_channels=input_dim,
            out_channels=output_dim,
            hidden_channels=hidden_dim_processor_node,
            edge_in_channels=2,
            edge_hidden_channels=hidden_dim_processor_edge,
            edge_out_channels=output_edge_dim,
            n_edge_layers=hidden_layers_processor_edge,
            n_node_layers=hidden_layers_processor_node
        )
        
    def forward(self, x):
        if x.dim() == 3:
            # unsqueeze time dim
            x = x.unsqueeze(1)
        b,t,n,f = x.size()
        
        batch_size = b * t
        squeezed_coarse_graph = einops.rearrange(x, 'b t n f -> (b t) n f')

        lat_lon_graph = einops.repeat(self.latlon_nodes, 'n f -> b n f', b=batch_size)

        decoded_graph, _ = self.coarse_to_fine((squeezed_coarse_graph, lat_lon_graph.to(x.device)), self.graph.edge_index.to(x.device), self.graph.edge_attr.to(x.device))
        return einops.rearrange(decoded_graph, '(b t) n f -> b t n f', b=b, t=t)
