# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
import torch
import xarray as xr

# from tsl.data.datamodule import SpatioTemporalDataModule
# from WeatherDataLoader import WeatherDataLoader
# from SpatioTemporalDataset import SpatioTemporalDataset
import torch_geometric
from models.DenseDCRNNToDiffConv import DenseDCRNNThenDiffConvModel
from models.TimeThenSpace import TimeThenSpaceModel
from models.TemporalSpikeGraphConvNet import TemporalSpikeGraphConvNet
from models.TemporalSynapticGraphConvNet import TemporalSynapticGraphConvNet
from models.TeporalSynapticLeanableWeights import TemporalSynapticLearnableWeights
from models.TSNStacked import TSNStacked
from models.PyGeoTemporalTest import A3TGCN
from models.GATDiffPooled import GATDiffPooled
from models.FAConvDiffPooled import FAConvDiffPooled
# from models.FAConvDiffPooled import GINonvDiffPooled
from models.DiffConvAttentionPool import GCNTemporalAttentionPooled
from models.DenseGCNMaxPooled import DenseGCNMaxPooled
from models.DenseConvDmon import DenseConvDmon
from models.TGatedGCN import TGatedGCN, TEdgeConv
from models.H3TGatedGCN import CoarseTGatedGCN
from models.MultiResGCN import MultiResGCN
from models.MLPStepThrough import MLPStepthrough
from models.AdaptiveGCN import AdaptiveGRUGCN
from models.DenseFiLMConv import DenseFiLMPlusConvNet
# from models.FaningDenseFiLM import FaningDenseFiLMP
from models.EchoStateFiLM import EchoStateFiLM
from models.GraphNBeats import GraphNBeats
from models.DynamicsModel import TemporalDynamics
from models.DenseFiLMWithLearnable import DenseFilmWithLearnable

from models.SynapticGCN import SynapticGCN
from models.MulristepPredictorRaw import MultistepPredictor
from models.SynapticAttention import SynapticAttention
from metrics.weighted_rmse2 import WeightedRMSE, _WeightedRMSE
from metrics.normalized_rmse import NormalizedRMSE
from metrics.RMSE import MaskedRMSE
from metrics.metric_utils import WeatherVariable


from tsl.nn.models import TransformerModel, GraphWaveNetModel
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from DataLoader import WeatherDL
from tsl.engines import Predictor
import warnings
import pickle
warnings.filterwarnings('ignore')


# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# from dask.distributed import Client

# client = Client(n_workers=4, threads_per_worker=1, memory_limit='30GB')
# print(client)

from load_data import get_data

z_min = torch.tensor([torch.tensor(-18.907263)]).expand(13)
q_min = torch.tensor([torch.tensor(-82.65108)]).expand(13)
t_min = torch.tensor([torch.tensor(-30.78354)]).expand(13)
u_min = torch.tensor([torch.tensor(-13.968123)]).expand(13)
v_min = torch.tensor([torch.tensor(-15.180508)]).expand(13)
tisr_min = torch.tensor([torch.tensor(-0.9964863)]).expand(13)

z_max = torch.tensor([torch.tensor(5.6104116)]).expand(13)
q_max = torch.tensor([torch.tensor(42.28271)]).expand(13)
t_max = torch.tensor([torch.tensor(9.658361)]).expand(13)
u_max = torch.tensor([torch.tensor(39.199768)]).expand(13)
v_max = torch.tensor([torch.tensor(15.876666)]).expand(13)
tisr_max = torch.tensor([torch.tensor(2.7472305)]).expand(13)

norm_scale = torch.stack([t_min, t_max])

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# alternative get_data which loads from multiple files, to avoid memory leak?
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
        # time="2003-06",
        # time=slice('1992', '2003'),
        # time=slice('2012', '2017'),
        time=slice('2010', '2017'),
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
        shuffle=True,
        normalization_range=normalization_range,
        # multiprocessing_context='fork'
        )
    # min, max = train.data_wrapper.getMinMaxValues()
    print('TRAIN DATALOADER DONE')
    valid = WeatherDL(
        data,
        # time='2004',
        # time='2004-05',
        time='2018',
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
        shuffle=True,
        normalization_range=normalization_range
        )
    print('VALIDATION DATALOADER DONE')
  

    input_size = train.spatio_temporal_dataset.n_channels   # n channel
    n_nodes = train.spatio_temporal_dataset.n_nodes         # n nodes
    horizon = train.spatio_temporal_dataset.horizon         # n prediction time steps
    print("DATA STUFF DONE")

    # hidden_size = 128   #@param
    learnable_size = 0   #@param
    rnn_layers = 1     #@param
    gnn_kernel = 2     #@param
    
    n_blocks = 3
    n_temporal_steps = 3
    
    prediction_horizon = 1
    n_layers = 20
    hidden_size = 256   #@param
    edge_embedding_size = 64
    spatial_block_size = 3
    temporal_block_size = 1
    lat_lons = train.data_wrapper.get_data().node.values

    model = DenseFilmWithLearnable(
        norm_scale=norm_scale,
        lat_lons=lat_lons,
        n_nodes=n_nodes,
        horizon=prediction_horizon,
        window=window,
        input_size=input_size,
        hidden_size=hidden_size,
        out_features=input_size,
        n_layers=n_layers,
        spatial_block_size=spatial_block_size,
        temporal_block_size=temporal_block_size,
    )

    # model = GraphWaveNetModel(
    #     input_size=input_size,
    #     output_size=input_size,
    #     horizon=prediction_horizon,
    #     hidden_size=hidden_size,
    #     ff_size=hidden_size,
    #     n_layers=n_layers,
    #     emb_size=hidden_size,
    #     n_nodes=n_nodes
    # )
    print('MODEL DONE')
    # model = TSNStacked(
    # input_size=input_size,
    # n_nodes=n_nodes,
    # horizon=horizon,
    # hidden_size=128,
    # output_type=output_type,
    # learnable_feature_size=32,
    # number_of_blocks=3,
    # number_of_temporal_steps=5
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
    # loss_fn = WeightedRMSE(weights, min_max=min_max, variables='all') #<-- currently running
    # loss_fn = MaskedRMSE(mask_inf=True, mask_nans=True)
    t = 0.3248
    z = 0.3014
    ts = torch.tensor([torch.tensor(t)]).expand(13)
    
    # loss_fn = _WeightedRMSE(weights,  variables='all', standardization=standardization, min_max=min_max)
    
    # feat_var_raw = train.data_wrapper.data.diff(dim='time').std('time').mean(dim=['level', 'lat', 'lon'])
    # z = feat_var_raw['z'].values
    # t = feat_var_raw['t'].values

    zs = torch.tensor([torch.tensor(z)]).expand(13)
    ts = torch.tensor([torch.tensor(t)]).expand(13)
    feature_variance = torch.cat([zs, ts])
    loss_fn = NormalizedRMSE(lat_lons=lat_lons, feature_variance=feature_variance, variables='all')
    
    metrics = {
        'weighted RMSE': WeightedRMSE(weights, standardization=standardization, min_max=min_max, variables=variables),
        'weighted RMSE denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, variables=variables, normalization_range=normalization_range),
        'weighted RMSE Z500': WeightedRMSE(weights, standardization=standardization, min_max=min_max, variables=[WeatherVariable('z', 500)]),
        'weighted RMSE T850': WeightedRMSE(weights, standardization=standardization, min_max=min_max, variables=[WeatherVariable('t', 850)]),
        'weighted RMSE Z500 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)]),
        'weighted RMSE T850 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)]),
        
        'weighted RMSE Z500 at day 1': WeightedRMSE(weights, standardization=standardization, min_max=min_max, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 1) - 1),
        'weighted RMSE T850 at day 1': WeightedRMSE(weights, standardization=standardization, min_max=min_max, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 1) - 1),
        
        # 'weighted RMSE Z500 at day 3': WeightedRMSE(weights, standardization=standardization, min_max=min_max, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 3) - 1),
        # 'weighted RMSE T850 at day 3': WeightedRMSE(weights, standardization=standardization, min_max=min_max, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 3) - 1),
        # 'weighted RMSE Z500 at day 5': WeightedRMSE(weights, standardization=standardization, min_max=min_max, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 5) - 1),
        # 'weighted RMSE T850 at day 5': WeightedRMSE(weights, standardization=standardization, min_max=min_max, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 5) - 1),
        'weighted RMSE Z500 at hour 6 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=0),
        'weighted RMSE T850 at hour 6 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=0),
        'weighted RMSE Z500 at hour 12 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=1),
        'weighted RMSE T850 at hour 12 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=1),
        'weighted RMSE Z500 at hour 18 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=2),
        'weighted RMSE T850 at hour 18 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=2),
        
        
        'weighted RMSE Z500 at day 1 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 1) - 1),
        'weighted RMSE T850 at day 1 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 1) - 1),
        # 'weighted RMSE Z500 at day 2 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 2) - 1),
        # 'weighted RMSE T850 at day 2 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 2) - 1),
        # 'weighted RMSE Z500 at day 3 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 3) - 1),
        # 'weighted RMSE T850 at day 3 denormalized': WeightedRMSE(weights, standardization=standardization, min_max=min_max, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 3) - 1),
        # 'weighted RMSE Z500 at day 5 denormalized': WeightedRMSE(weights, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 5) - 1),
        # 'weighted RMSE T850 at day 5 denormalized': WeightedRMSE(weights, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 5) - 1),
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
    
    
    # checkpoint_path = 'checkpoint/DenseFiLMPlusConvNet_8.3324-v84.ckpt' #20 layer version
    # checkpoint_path = 'checkpoint/DenseFiLMPlusConvNet_8.3324-v85.ckpt' #30 layer version
    # checkpoint_path = 'checkpoint/DenseFilmWithLearnable_8.3324-v5.ckpt' #20 layer version
    checkpoint_path = 'checkpoint/DenseFilmWithLearnable2_8.3324.ckpt' #20 layer version
    
    # torch_checkpoint = torch.load(checkpoint_path)
    
    predictor = MultistepPredictor(
        model=model,                   # our initialized model
        # model=compiled_model,
        # optim_class=torch.optim.Adam,  # specify optimizer to be used...
        # optim_kwargs={'lr': 0.001},    # ...and parameters for its initialization
        loss_fn=loss_fn,               # which loss function to be used
        metrics=metrics,               # metrics to be logged during train/val/test
        prediction_horizon=prediction_horizon,
        sequence_horizon=horizon,
        accumulate_grad_batches=8,
        initial_rollouts=1,
        rollout_epoch_breakpoint=100,
    )
    # predictor.load_model(filename=checkpoint_path)
    # predictor.automatic_optimization=False
    # predictor._set_multistep_attrs(4, horizon)
    # predictor.set_metrics(metrics)

    print(device,model, torch.cuda.device_count(), model._get_name(), torch_geometric.__version__)

    version = 8.3324 #332 - using learned adj, not learning
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
    # 8.16 synaptic stuff
    # 8.17 attention for all timesteps
    # 8.21 MSE 
    # 8.22 no dense convs, just more propagations per existing spatial graph (5)
    # 8.23 back to GRU, conditioning initial H on last observed timestep, GRU with GATE gates
    # 8.24 no conditioning residual orig X, all variables RMSE
    logger = TensorBoardLogger(save_dir=f"logs/{model._get_name()}", name=f"not overfit DiffConv comp {model._get_name()}2 hidden layer size = [{hidden_size}], temporal steps = {n_temporal_steps}, learnable size = {learnable_size} time - resolution: {temporal_resolution}, window: {window}, horizon: {horizon}, prediction_horizon: {prediction_horizon} n_layers: {n_layers}, spatial_block_size={spatial_block_size}", version=version)


    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint',
        save_top_k=1,
        monitor='val_weighted RMSE',
        mode='min',
        filename=f"{model._get_name()}2_{version}"
    )
    
    early_stopping = EarlyStopping(monitor='val_weighted RMSE', mode='min', min_delta=1e-5,patience=3, check_on_train_epoch_end=True)
    
    SWA = StochasticWeightAveraging(1e2, 0.2,5)
    # compiled_predictor = torch.compile(predictor, mode='reduce-overhead')

    
    trainer = pl.Trainer(max_epochs=10,
                        logger=logger,
                        # logger=False,
                        #  gpus=1 if torch.cuda.is_available() else None,
                        devices=1,
                        accelerator='gpu',
                        #  limit_train_batches=8000, #n_batches - 1 epoch, for easier scheduling and that sort of thing?
                        # limit_val_batches=5000, 
                        # callbacks=[checkpoint_callback, early_stopping, SWA],
                        callbacks=[checkpoint_callback],
                        # detect_anomaly=True,
                        # accumulate_grad_batches=32,
                        # val_check_interval=0.25,
                        # limit_val_batches=0.50
                        # num_sanity_val_steps=1,
                        )


    # trainer.fit(predictor, train_dataloaders=train.data_loader, val_dataloaders=valid.data_loader, ckpt_path=checkpoint_path)
    trainer.fit(predictor, train_dataloaders=train.data_loader, val_dataloaders=valid.data_loader)
    
    torch.save(model, f'checkpoint/plain_model_multistep_{model._get_name()}2_{version}')

if __name__ == '__main__':
    # medium, hight, highest
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed=42)
    train()
    
    """
    other ideas 
        - GNN with n message propagation steps
        - superconvergence (LR scheduling)
        - MLP for result generation (faster cycles? less memory? more blocks?)
        - gru with non shit activations for readout?
        - attention based readout?(need to collapse time into less timesteps + include initial data as exogenous variables)
    """