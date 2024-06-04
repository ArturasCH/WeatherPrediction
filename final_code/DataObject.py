# A modified Data object from TSL library to work with xarray data
# dependency of SpatiotemporalDataset

import torch
import xarray as xr
from torch import Tensor
from torch_sparse import SparseTensor
from tsl.data.data import Data as TslData, StorageView as TslStorageView
from torch_geometric.typing import Adj
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping,
                    Optional, Tuple, Union)

def pattern_size_repr(key: str,
                      x: Union[Tensor, SparseTensor, xr.core.dataarray.DataArray],
                      pattern: str = None):
    if isinstance(x, xr.core.dataarray.DataArray):
        x = torch.Tensor(x.load().data)
    if pattern is not None:
        pattern = pattern.replace(' ', '')
        out = str([
            f'{token}={size}' if not token.isnumeric() else str(size)
            for token, size in zip(pattern, get_size(x))
        ])
    else:
        out = str(list(get_size(x)))
    out = f"{key}={out}".replace("'", '')
    return out

def get_size(x: Union[Tensor, SparseTensor]) -> Tuple:
    if isinstance(x, Tensor):
        return tuple(x.size())
    elif isinstance(x, SparseTensor):
        return tuple(x.sizes())
    
class StorageView(TslStorageView):
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [pattern_size_repr(k, v) for k, v in self.items()]
        return '{}({})'.format(cls, ', '.join(info))
    
class Data(TslData):
    def __init__(self,
                 input: Optional[Mapping] = None,
                 target: Optional[Mapping] = None,
                 edge_index: Optional[Adj] = None,
                 edge_weight: Optional[Tensor] = None,
                 mask: Optional[Tensor] = None,
                 transform: Optional[Mapping] = None,
                 pattern: Optional[Mapping] = None,
                 **kwargs):
        input = input if input is not None else dict()
        target = target if target is not None else dict()
        super(Data, self).__init__(**input,
                                   **target,
                                   edge_index=edge_index,
                                   edge_weight=edge_weight,
                                   **kwargs)
        # Set 'input' as view on input keys
        self.__dict__['input'] = StorageView(self._store, input.keys())
        # Set 'target' as view on target keys
        self.__dict__['target'] = StorageView(self._store, target.keys())
        # Add mask
        self.mask = mask  # noqa
        # Add transform modules
        transform = transform if transform is not None else dict()
        self.transform: dict = transform  # noqa
        # Add patterns
        self.__dict__['pattern'] = dict()
        if pattern is not None:
            self.pattern.update(pattern)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        inputs = [
            pattern_size_repr(k, v, self.pattern.get(k))
            for k, v in self.input.items()
        ]
        inputs = 'input=({})'.format(', '.join(inputs))
        targets = [
            pattern_size_repr(k, v, self.pattern.get(k))
            for k, v in self.target.items()
        ]
        targets = 'target=({})'.format(', '.join(targets))
        info = [inputs, targets, "has_mask={}".format(self.has_mask)]
        if self.has_transform:
            info += ["transform=[{}]".format(', '.join(self.transform.keys()))]
        return '{}(\n  {}\n)'.format(cls, ',\n  '.join(info))
    
    
    