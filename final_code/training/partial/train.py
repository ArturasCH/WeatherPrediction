# TGNN partial training loop

import torch
import xarray as xr

from model.DenseFiLMWithLearnable import DenseFilmWithLearnable

from PartialMulristepTrainer import MultistepPredictor
from ..metrics.normalized_mse import NormalizedMSE
from ..metrics.weighted_rmse2 import WeightedRMSE
from metrics.metric_utils import WeatherVariable

from lightning.pytorch.loggers import TensorBoardLogger
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from data.DataLoader import WeatherDLPartial
import warnings
import pickle
warnings.filterwarnings('ignore')

from data.load_data import get_data

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    data = get_data()
    batch_size = 4
    num_workers = 6
    prefetch_factor = 1
    persistent_workers=True
    pin_memory = False
    temporal_resolution = 6
    window = 20
    horizon = 4
    train = WeatherDLPartial(
        data,
        # time=slice('1992', '2003'),
        # time=slice('2012', '2017'),
        time=slice('2010', '2017'),
        temporal_resolution=temporal_resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        window=window,
        horizon=horizon,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=True,
        )
    print('TRAIN DATALOADER DONE')
    valid = WeatherDLPartial(
        data,
        time='2018',
        temporal_resolution=temporal_resolution,
        horizon=4,
        # mean=standardization['mean'],
        # std=standardization['std'],
        batch_size=batch_size,
        num_workers=num_workers,
        window=window,
        horizon=horizon,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=True,
        # 
        )
    print('VALIDATION DATALOADER DONE')
  

    input_size = train.spatio_temporal_dataset.n_channels   # n channel
    n_nodes = train.spatio_temporal_dataset.n_nodes         # n nodes
    horizon = train.spatio_temporal_dataset.horizon         # n prediction time steps
    print("DATA STUFF DONE")
    
    prediction_horizon = 1
    n_layers = 20
    hidden_size = 256   #@param
    edge_embedding_size = 64
    spatial_block_size = 3
    temporal_block_size = 1
    lat_lons = train.data_wrapper.get_data().node.values

    model = DenseFilmWithLearnable(
        # norm_scale=norm_scale,
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

    steps_per_day = 24 // temporal_resolution
    
    variables = [
        WeatherVariable('z', 500),
        WeatherVariable('t', 850)
        ]
    weights = train.data_wrapper.node_weights
    t = 0.3248
    z = 0.3014

    zs = torch.tensor([torch.tensor(z)]).expand(13)
    ts = torch.tensor([torch.tensor(t)]).expand(13)
    feature_variance = torch.cat([zs, ts])
    loss_fn = NormalizedMSE(lat_lons=lat_lons, feature_variance=feature_variance, variables='all')
    
    metrics = {
        'weighted RMSE': WeightedRMSE(weights, variables=variables),
        'weighted RMSE denormalized': WeightedRMSE(weights, denormalize=True, variables=variables),
        'weighted RMSE Z500': WeightedRMSE(weights, variables=[WeatherVariable('z', 500)]),
        'weighted RMSE T850': WeightedRMSE(weights, variables=[WeatherVariable('t', 850)]),
        'weighted RMSE Z500 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('z', 500)]),
        'weighted RMSE T850 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('t', 850)]),
        
        'weighted RMSE Z500 at day 1': WeightedRMSE(weights, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 1) - 1),
        'weighted RMSE T850 at day 1': WeightedRMSE(weights, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 1) - 1),

        'weighted RMSE Z500 at hour 6 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('z', 500)], at=0),
        'weighted RMSE T850 at hour 6 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('t', 850)], at=0),
        'weighted RMSE Z500 at hour 12 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('z', 500)], at=1),
        'weighted RMSE T850 at hour 12 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('t', 850)], at=1),
        'weighted RMSE Z500 at hour 18 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('z', 500)], at=2),
        'weighted RMSE T850 at hour 18 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('t', 850)], at=2),
        
        
        'weighted RMSE Z500 at day 1 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 1) - 1),
        'weighted RMSE T850 at day 1 denormalized': WeightedRMSE(weights, denormalize=True, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 1) - 1),
    }
    checkpoint_path = 'checkpoint/DenseFilmWithLearnable2_8.3324.ckpt' #20 layer version
    
    
    predictor = MultistepPredictor(
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        prediction_horizon=prediction_horizon,
        sequence_horizon=horizon,
        accumulate_grad_batches=8,
        initial_rollouts=1,
        rollout_epoch_breakpoint=100,
    )

    print(device,model, torch.cuda.device_count(), model._get_name())

    version = 8.3324
    logger = TensorBoardLogger(
        save_dir=f"logs/{model._get_name()}",
        name=f"not overfit DiffConv comp {model._get_name()}2 hidden layer size = [{hidden_size}], time - resolution: {temporal_resolution}, window: {window}, horizon: {horizon}, prediction_horizon: {prediction_horizon} n_layers: {n_layers}, spatial_block_size={spatial_block_size}", version=version
        )


    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint',
        save_top_k=1,
        monitor='val_weighted RMSE',
        mode='min',
        filename=f"{model._get_name()}2_{version}"
    )
    

    
    trainer = pl.Trainer(max_epochs=10,
                        logger=logger,
                        devices=1,
                        accelerator='gpu',
                        callbacks=[checkpoint_callback],
                        )


    # trainer.fit(predictor, train_dataloaders=train.data_loader, val_dataloaders=valid.data_loader, ckpt_path=checkpoint_path)
    trainer.fit(predictor, train_dataloaders=train.data_loader, val_dataloaders=valid.data_loader)
    
    torch.save(model, f'checkpoint/plain_model_multistep_{model._get_name()}2_{version}')

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed=42)
    train()
    