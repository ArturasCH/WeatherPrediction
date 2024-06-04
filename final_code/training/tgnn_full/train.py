import torch
import xarray as xr
import torch_geometric
from models.FiLMWithLearnable import FiLMWithLearnable

from MultistepSeriesTrainer import MultistepTrainer
from ..metrics.weighted_rmse import WeightedRMSE
from ..metrics.metric_utils import WeatherVariable

from lightning.pytorch.loggers import TensorBoardLogger
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from data.DataLoader import WeatherDLFull
import warnings
import pickle
warnings.filterwarnings('ignore')

from data.load_data import get_data

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = get_data()
    mean_std = pickle.load(open('../data/data_files/mean_std_stacked.pickle', 'rb'))
    batch_size = 8
    num_workers = 6
    prefetch_factor = 1
    persistent_workers=True
    pin_memory = False
    temporal_resolution = 6
    rollouts = 2
    window = 20
    sequence_horizon = 4
    horizon = 0
    train = WeatherDLFull(
        data,
        # time="2003-06",
        # time=slice('1992', '2003'),
        # time=slice('2012', '2017'),
        time=slice('2013', '2017'),
        temporal_resolution=temporal_resolution,
        mean=mean_std['mean'],
        std=mean_std['std'],
        batch_size=batch_size,
        num_workers=num_workers,
        window=window + rollouts,
        horizon=horizon,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=True,
        )

    print('TRAIN DATALOADER DONE')
    valid = WeatherDLFull(
        data,
        time='2018',
        temporal_resolution=temporal_resolution,
        mean=mean_std['mean'],
        std=mean_std['std'],
        batch_size=batch_size,
        num_workers=num_workers,
        window=window + sequence_horizon,
        horizon=horizon,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=True,
        )
    print('VALIDATION DATALOADER DONE')
  

    input_size = train.spatio_temporal_dataset.n_channels   # n channel
    n_nodes = train.spatio_temporal_dataset.n_nodes         # n nodes

    learnable_size = 32
    
    prediction_horizon = 1
    n_layers = 20
    hidden_size = 256
    spatial_block_size = 3
    lat_lons = train.data_wrapper.get_data().node.values

    constants = xr.open_dataset('./data/data_files/constants_5.625deg.nc').stack(node=['lat', 'lon'])
    lsm = torch.tensor(constants['lsm'].data)
    model = FiLMWithLearnable(
        lsm=lsm,
        lat_lons=lat_lons,
        n_nodes=n_nodes,
        input_size=input_size,
        hidden_size=hidden_size,
        out_features=input_size,
        n_layers=n_layers,
        window=window,
        horizon=1,
        spatial_block_size=spatial_block_size,
    )

    steps_per_day = 24 // temporal_resolution
    variables = [
        WeatherVariable('z', 500),
        WeatherVariable('t', 850)
        ]
    weights = train.data_wrapper.node_weights
    standardization = {
        'mean': train.data_wrapper.mean,
        'std': train.data_wrapper.std
    }
    loss_fn = WeightedRMSE(weights, standardization=standardization, variables='all')
    metrics = {
        'weighted RMSE': WeightedRMSE(weights, standardization=standardization, variables=variables),
        'weighted RMSE denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, variables=variables, normalization_range=normalization_range),
        'weighted RMSE Z500': WeightedRMSE(weights, standardization=standardization, variables=[WeatherVariable('z', 500)]),
        'weighted RMSE T850': WeightedRMSE(weights, standardization=standardization, variables=[WeatherVariable('t', 850)]),
        'weighted RMSE Z500 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)]),
        'weighted RMSE T850 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)]),
        
        'weighted RMSE Z500 at day 1': WeightedRMSE(weights, standardization=standardization, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 1) - 1),
        'weighted RMSE T850 at day 1': WeightedRMSE(weights, standardization=standardization, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 1) - 1),
        
        'weighted RMSE Z500 at hour 6 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=0),
        'weighted RMSE T850 at hour 6 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=0),
        'weighted RMSE Z500 at hour 12 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=1),
        'weighted RMSE T850 at hour 12 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=1),
        'weighted RMSE Z500 at hour 18 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=2),
        'weighted RMSE T850 at hour 18 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=2),
        
        
        'weighted RMSE Z500 at day 1 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 1) - 1),
        'weighted RMSE T850 at day 1 denormalized': WeightedRMSE(weights, standardization=standardization, denormalize=True, normalization_range=normalization_range, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 1) - 1),
    }
    

    checkpoint_path = 'checkpoint/FiLMWithLearnable_8.3324_256_rollouts_3_layers_20.ckpt' #20 layer version
    
    
    multistep_trainer = MultistepTrainer(
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        prediction_horizon=prediction_horizon,
        sequence_horizon=sequence_horizon,
        window=window,
        accumulate_grad_batches=4,
        initial_rollouts=rollouts,
        rollout_epoch_breakpoint=100,
    )
    print(device,model, torch.cuda.device_count(), model._get_name(), torch_geometric.__version__)

    version = 8.3324
    logger = TensorBoardLogger(save_dir=f"logs/{model._get_name()}", name=f"not overfit seq DiffConv orig {model._get_name()} hidden layer size = [{hidden_size}], learnable size = {learnable_size} time - resolution: {temporal_resolution}, prediction_horizon: {prediction_horizon} n_layers: {n_layers}, spatial_block_size={spatial_block_size}", version=version)


    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint',
        save_top_k=2,
        monitor='val_weighted RMSE',
        mode='min',
        filename=f"{model._get_name()}_{version}_{hidden_size}_rollouts_{rollouts}_layers_{n_layers}"
    )
    
    trainer = pl.Trainer(max_epochs=100,
                        logger=logger,
                        devices=1,
                        accelerator='gpu',
                        callbacks=[checkpoint_callback],
                        )


    trainer.fit(multistep_trainer, train_dataloaders=train.data_loader, val_dataloaders=valid.data_loader, ckpt_path=checkpoint_path)
    # trainer.fit(predictor, train_dataloaders=train.data_loader, val_dataloaders=valid.data_loader)
    

if __name__ == '__main__':
    # medium, hight, highest
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed=42)
    train()