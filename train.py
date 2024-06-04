import torch
import xarray as xr
# from tsl.data.datamodule import SpatioTemporalDataModule
# from WeatherDataLoader import WeatherDataLoader
# from SpatioTemporalDataset import SpatioTemporalDataset

from models.DenseDCRNNToDiffConv import DenseDCRNNThenDiffConvModel
from models.TimeThenSpace import TimeThenSpaceModel
from models.TemporalSpikeGraphConvNet import TemporalSpikeGraphConvNet
from models.TemporalSynapticGraphConvNet import TemporalSynapticGraphConvNet
from models.TeporalSynapticLeanableWeights import TemporalSynapticLearnableWeights
from models.TSNStacked import TSNStacked
from models.TSNStacked2 import TSNStacked2
from models.TSNStacked3 import TSNStacked3
from models.PyGeoTemporalTest import A3TGCN
from models.AdjPretrain import AdjPretrain
from models.TFiLM import TMixHop
from models.TGatedGCN import TGatedGCN
from models.TPNA import TPNA
from models.DenseFiLMConv import DenseFiLMPlusConvNet

from models.SynapticGCN import SynapticGCN
from models.MultistepPredictor import MultistepPredictor
from models.SynapticAttention import SynapticAttention
from metrics.weighted_rmse2 import WeightedRMSE
from metrics.metric_utils import WeatherVariable

from tsl.nn.models import TransformerModel, GraphWaveNetModel
from tsl.metrics.torch import MaskedMSE
from tsl.engines import Predictor
from lightning.pytorch.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from DataLoader import WeatherDL
import warnings
import pickle
warnings.filterwarnings('ignore')


# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# from dask.distributed import Client

# client = Client(n_workers=4, threads_per_worker=1, memory_limit='30GB')
# print(client)

from load_data import get_data

from torch_geometric.utils import degree
def get_degree_bins(edge_index, n_nodes):
    max_degree = -1
    d = degree(edge_index[1], num_nodes=n_nodes, dtype=torch.long)
    max_degree = max(max_degree, int(d.max()))
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    d = degree(edge_index[1], num_nodes=n_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())
    return deg

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = get_data()
    # train_loader = WeatherDataLoader(data, time=slice('2000', '2002'), temporal_resolution=3)
    # min, max = train_loader.getMinMaxValues()
    # validation_loader = WeatherDataLoader(data, time='2003', temporal_resolution=3, min=min, max=max)
    # min_max = pickle.load(open('./min_max_test.pkl', 'rb'))
    min_max = pickle.load(open('./min_max_train_full.pkl', 'rb'))
    standardization = pickle.load(open('./standardization.pkl', 'rb'))
    sr_standardization = pickle.load(open('./sr_standardization.pkl', 'rb'))
    standardization['mean'] = xr.concat([standardization['mean'],sr_standardization['mean']], 'level')
    standardization['std'] = xr.concat([standardization['std'],sr_standardization['std']], 'level')
    # min_max = pickle.load(open('./min_max_global.pkl', 'rb'))
    normalization_range = {'min': -1, 'max': 1}
    batch_size = 4
    num_workers = 6
    prefetch_factor = 1
    persistent_workers=True
    pin_memory = False
    temporal_resolution = 6
    window = 20
    horizon = 4
    train = WeatherDL(
        data,
        time="2003-06",
        # time=slice('2001', '2003'),
        temporal_resolution=temporal_resolution,
        min=min_max['min'],
        max=min_max['max'],
        mean=standardization['mean'],
        std=standardization['std'],
        batch_size=batch_size,
        num_workers=num_workers,
        window=window,
        horizon=horizon,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=False,
        normalization_range=normalization_range,
        # multiprocessing_context='fork'
        )
    # min, max = train.data_wrapper.getMinMaxValues()
    print('TRAIN DATALOADER DONE')
    valid = WeatherDL(
        data,
        time='2004-05',
        temporal_resolution=temporal_resolution,
        # horizon=8,
        min=min_max['min'],
        max=min_max['max'],
        mean=standardization['mean'],
        std=standardization['std'],
        batch_size=batch_size,
        num_workers=num_workers,
        window=window,
        horizon=horizon,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=False,
        normalization_range=normalization_range
        # multiprocessing_context='fork'
        )
    print('VALIDATION DATALOADER DONE')

    # import pandas as pd
    # data = train_loader.get_data()
    # time_index = pd.DatetimeIndex(data.time,freq='infer')
    # from tsl.data.preprocessing import StandardScaler, Scaler, ScalerModule, MinMaxScaler
    # # ScalerModule(scaler=MinMaxScaler())

    # torch_dataset = SpatioTemporalDataset(target=train_loader.get_data(),
    #                                     connectivity=train_loader.get_connectivity(),
    #                                     #   index=time_index,
    #                                     horizon=40,
    #                                     window=112,
    #                                     stride=1
    #                                     )
    # # test_dataset = SpatioTemporalDataset(target=test_loader.get_data(),
    # #                                       connectivity=test_loader.get_connectivity(),
    # #                                       horizon=40,
    # #                                       window=112,
    # #                                       stride=1)
    # validation_dataset = SpatioTemporalDataset(target=validation_loader.get_data(),
    #                                     connectivity=validation_loader.get_connectivity(),
    #                                     horizon=40,
    #                                     window=112,
    #                                     stride=1)

    # train_dm = SpatioTemporalDataModule(
    #     dataset=torch_dataset,
    #     # scalers=scalers,
    #     # splitter=splitter,
    #     batch_size=5,
    #     # workers=8,
    #     pin_memory=True,
    # )
    # test_dm = SpatioTemporalDataModule(
    #     dataset=test_dataset,
    #     scalers=scalers,
    #     splitter=splitter,
    #     batch_size=50,
    #     # workers=1
    # )
    # validation_dm = SpatioTemporalDataModule(
    #     dataset=validation_dataset,
    #     # scalers=scalers,
    #     # splitter=splitter,
    #     batch_size=5,
    #     pin_memory=True,
    #     # workers=4
    # )
    # train_dm.setup()
    # validation_dm.setup()

    input_size = train.spatio_temporal_dataset.n_channels   # n channel
    n_nodes = train.spatio_temporal_dataset.n_nodes         # n nodes
    horizon = train.spatio_temporal_dataset.horizon         # n prediction time steps


    hidden_size = 256   #@param
    # hidden_size = 128   #@param
    learnable_size = 32   #@param
    rnn_layers = 15     #@param
    gnn_kernel = 15     #@param


    # model = TimeThenSpaceModel(input_size=input_size,
    #                            n_nodes=n_nodes,
    #                            horizon=horizon,
    #                            hidden_size=hidden_size,
    #                            rnn_layers=rnn_layers,
    #                            gnn_kernel=gnn_kernel)

    # model = TransformerModel(
    #     input_size=input_size,
    #     output_size=input_size,
    #     horizon=horizon,
    #     hidden_size=64,
    #     ff_size=64,
    #     n_heads=2,
    #     n_layers=1,
    #     axis='both')

    # model = DenseDCRNNThenDiffConvModel(
    #     input_size,
    #     n_nodes=n_nodes,
    #     horizon=horizon,
    #     temporal_layers=5,
    #     adjacency=train.data_wrapper.adjacency_weighted,
    #     device=device
    #     )

    # model = GraphWaveNetModel(
    #     input_size=input_size,
    #     output_size=input_size,
    #     horizon=horizon,
    #     n_nodes=n_nodes,
    #     hidden_size=hidden_size,
    #     emb_size=hidden_size,
    #     spatial_kernel_size=3,
    #     )
    
    # model = TemporalSpikeGraphConvNet(
    #                            input_size=input_size,
    #                            n_nodes=n_nodes,
    #                            horizon=horizon,
    #                            hidden_size=hidden_size * 8,
    #                            use_spike_for_output=True
    #                            )
    # output_type = "membrane_potential"
    # val_weighted RMSE=0.163, val_weighted RMSE T850=0.150, val_weighted RMSE T850 at day 3=0.178, val_weighted RMSE Z500=0.177, val_weighted RMSE Z500 at day 3=0.169, train_weighted RMSE=0.203, train_weighted RMSE T850=0.213, train_weighted RMSE T850 at day 3
    # model = TemporalSynapticGraphConvNet(
    #     input_size=input_size,
    #     n_nodes=n_nodes,
    #     horizon=horizon,
    #     hidden_size=128,
    #     # horizon=8,5
    #     output_type=output_type,
    #     number_of_blocks=8,
    #     number_of_temporal_steps=3
    # )
    
    # val_weighted RMSE=0.173, val_weighted RMSE T850=0.155, val_weighted RMSE T850 at day 3=0.150, val_weighted RMSE Z500=0.191, val_weighted RMSE Z500 at day 3=0.177, train_weighted RMSE=0.219, train_weighted RMSE T850=0.243, train_weighted RMSE T850 at day 3
    
    # CKPT_PATH = "checkpoint/A3TGCN_8.15.ckpt"
    # model = A3TGCN(
    #     input_size=input_size,
    #     n_nodes=n_nodes,
    #     # horizon=horizon,
    #     horizon=4,
    #     hidden_size=hidden_size,
    #     # output_type=output_type,
    #     number_of_blocks=n_blocks,
    #     # number_of_temporal_steps=n_temporal_steps,
    #     # edge_index=train.data_wrapper.edge_index.to(device),
    #     # edge_weight=train.data_wrapper.edge_weight.to(device)
    #     window=window,
    #     learnable_feature_size=learnable_size
    # )
    prediction_horizon =4
    n_layers = 12
    edge_embedding_size = 64
    spatial_block_size = 2
    temporal_block_size = 1
    
    # edge_index, edge_attr = train.data_wrapper.get_connectivity()
    # deg = get_degree_bins(edge_index, n_nodes)
    
    model = DenseFiLMPlusConvNet(
        # deg=deg,
        lat_lons=train.data_wrapper.get_data().node.values,
        n_nodes=n_nodes,
        horizon=prediction_horizon,
        window=window,
        input_size=input_size,
        hidden_size=hidden_size,
        out_features=input_size,
        n_layers=n_layers,
        spatial_block_size=spatial_block_size,
        temporal_block_size=temporal_block_size
        # n_relations=6
    )

    # model = TSNStacked3(
    # input_size=input_size,
    # n_nodes=n_nodes,
    # horizon=horizon,
    # hidden_size=hidden_size,
    # output_type=output_type,
    # learnable_feature_size=learnable_size,
    # number_of_blocks=spatial_block_size,
    # number_of_temporal_steps=n_temporal_steps
    # )
    
    # model = SynapticAttention(
    #     input_size=input_size,
    #     n_nodes=n_nodes,
    #     horizon=horizon,
    #     hidden_size=hidden_size,
    #     number_of_blocks=3
    # )
    # model = SynapticGCN(
    #    input_size=input_size,
    #     n_nodes=n_nodes,
    #     horizon=horizon,
    #     hidden_size=hidden_size * 5,
    #     output_type="spike",
    #     number_of_blocks=3,
    #     prediction_horizon=8
    # )

    # loss_fn = MaskedMSE()
    steps_per_day = 24 // temporal_resolution
    # metrics = {'mse': MaskedMSE(),
    #         #    'mape': MaskedMAPE(),
    #         'mse_at_3_days': MaskedMSE(at=(steps_per_day * 3) - 1),  # 'steps_per_day * 3' indicates the 24th time step,
    #                                         # which correspond to 3 days ahead
    #         'mse_at_5_days': MaskedMSE(at=(steps_per_day * 5) - 1)
    #         }

    variables = [
        # WeatherVariable('z', 500),
        WeatherVariable('t', 850)
        ]
    weights = train.data_wrapper.node_weights
    # loss_fn = WeightedRMSE(weights, min_max=min_max, variables='all')
    # loss_fn = MaskedMSE()
    loss_fn = WeightedRMSE(weights, min_max=min_max, variables='all', standardization=standardization)
    metrics = {
        'weighted RMSE': WeightedRMSE(weights,min_max=min_max, standardization=standardization, variables=variables),
        'weighted RMSE denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, variables=variables, normalization_range=normalization_range),
        # 'weighted RMSE Z500': WeightedRMSE(weights,min_max=min_max, standardization=standardization, variables=[WeatherVariable('z', 500)]),
        'weighted RMSE T850': WeightedRMSE(weights,min_max=min_max, standardization=standardization, variables=[WeatherVariable('t', 850)]),
        # 'weighted RMSE Z500 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)]),
        'weighted RMSE T850 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)]),
        
        # 'weighted RMSE Z500 at day 1': WeightedRMSE(weights,min_max=min_max, standardization=standardization, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 1) - 1),
        # 'weighted RMSE T850 at day 1': WeightedRMSE(weights,min_max=min_max, standardization=standardization, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 1) - 1),
        
        # 'weighted RMSE Z500 at day 3': WeightedRMSE(weights,min_max=min_max, standardization=standardization, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 3) - 1),
        # 'weighted RMSE T850 at day 3': WeightedRMSE(weights,min_max=min_max, standardization=standardization, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 3) - 1),
        # 'weighted RMSE Z500 at day 5': WeightedRMSE(weights,min_max=min_max, standardization=standardization, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 5) - 1),
        # 'weighted RMSE T850 at day 5': WeightedRMSE(weights,min_max=min_max, standardization=standardization, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 5) - 1),
        # 'weighted RMSE Z500 at hour 6 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=0),
        'weighted RMSE T850 at hour 6 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=0),
        # 'weighted RMSE Z500 at hour 12 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=1),
        'weighted RMSE T850 at hour 12 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=1),
        # 'weighted RMSE Z500 at hour 18 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=2),
        'weighted RMSE T850 at hour 18 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=2),
        
        
        # 'weighted RMSE Z500 at day 1 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 1) - 1),
        'weighted RMSE T850 at day 1 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 1) - 1),
        # 'weighted RMSE Z500 at day 2 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 2) - 1),
        # 'weighted RMSE T850 at day 2 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 2) - 1),
        # 'weighted RMSE Z500 at day 3 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 3) - 1),
        # 'weighted RMSE T850 at day 3 denormalized': WeightedRMSE(weights,min_max=min_max, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 3) - 1),
        # 'weighted RMSE Z500 at day 5 denormalized': WeightedRMSE(weights,min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 5) - 1),
        # 'weighted RMSE T850 at day 5 denormalized': WeightedRMSE(weights,min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 5) - 1),
    }
    # loss_fn = MaskedRMSE()
    # steps_per_day = 24 // 3
    # metrics = {'rmse': MaskedRMSE(),
    #         #    'mape': MaskedMAPE(),
    #            'rmse_at_3_days': MaskedRMSE(at=(steps_per_day * 3) - 1),  # 'steps_per_day * 3' indicates the 24th time step,
    #                                           # which correspond to 3 days ahead
    #            'rmse_at_5_days': MaskedRMSE(at=(steps_per_day * 5) - 1)}

    # setup predictor
    # stgnn = stgnn.to(torch.device(device))
    # compiled_model = torch.compile(model, mode='reduce-overhead')
    
    predictor = Predictor(
        model=model,                   # our initialized model
        # model=compiled_model,
        optim_class=torch.optim.Adam,  # specify optimizer to be used...
        optim_kwargs={'lr': 0.001},    # ...and parameters for its initialization
        loss_fn=loss_fn,               # which loss function to be used
        metrics=metrics,                # metrics to be logged during train/val/test
        # scheduler_class=torch.optim.lr_scheduler.CyclicLR,
        # scheduler_kwargs={'patience':5, 'cooldown': 2, 'monitor': 'train_weighted RMSE_epoch'},
        # scheduler_kwargs={'max_lr':0.01, 'base_lr': 1e-8, 'step_size_up': 30, 'step_size_down': 30,'cycle_momentum': False},
    )
    # predictor._set_multistep_attrs(8, horizon)
    # predictor.load_model(filename=CKPT_PATH)

    # print(device,model, torch.cuda.device_count(), model._get_name())

    version = 8.3324
    # 7.4 - blocks 4/temporal 5/ learnable 48/hidden 128
    # 7.5 - blocks 5/temporal 5/ learnable 48/hidden 128
    # 7.6 - blocks 5/temporal 5/ learnable 64/hidden 128 adamW
    # 7.7 - blocks 5/temporal 5/ learnable 64/hidden 128 adam
    # 7.7 - blocks 4/temporal 5/ learnable 48/hidden 256 adam
    # 8.3 - has scheduler 100/0.1
    # 8.5 layer norm
    # 8.6 instance norm
    # 8.7 no node embeddings
    # 8.9 no learned adj, batch norm, bigger readout hidden state
    # 8.13 1 day prediction, transformer time
    # 8.14 1 day MSE loss, synaptic time
    logger = TensorBoardLogger(save_dir=f"logs/{model._get_name()}", name=f"Overfit /w learnable/film out v2/CLRS: {model._get_name()} film skip no reorder, hidden_size: {hidden_size} layers:{n_layers} spatial_block_size = {spatial_block_size}, temporal_block_size = {temporal_block_size}, time - resolution: {temporal_resolution}, window: {window}, horizon: {horizon}", version=version)


    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint',
        save_top_k=1,
        monitor='val_weighted RMSE',
        mode='min',
        filename=f"{model._get_name()}_{version}"
    )
    
    early_stopping = EarlyStopping(monitor='train_weighted RMSE_epoch', mode='min', min_delta=1e-5,patience=50, check_on_train_epoch_end=True)
    
    # compiled_predictor = torch.compile(predictor, mode='reduce-overhead')

    trainer = pl.Trainer(max_epochs=200,
                        logger=logger,
                        # logger=False,
                        #  gpus=1 if torch.cuda.is_available() else None,
                        devices=1,
                        accelerator='gpu',
                        #  limit_train_batches=150,
                        # limit_val_batches=50,
                        callbacks=[early_stopping],
                        # accumulate_grad_batches=8,
                        # val_check_interval=320,
                        # limit_val_batches=0.50,
                        # num_sanity_val_steps=0,
                        gradient_clip_val=0.5,
                        )


    trainer.fit(predictor, train_dataloaders=train.data_loader, val_dataloaders=valid.data_loader)
    
    # torch.save(model, f'checkpoint/plain_model_{model._get_name()}_{version}')
    

if __name__ == '__main__':
    # torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed=42)
    train()