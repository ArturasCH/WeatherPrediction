import torch
# from tsl.data.datamodule import SpatioTemporalDataModule
# from WeatherDataLoader import WeatherDataLoader
# from SpatioTemporalDataset import SpatioTemporalDataset

from models.DenseDCRNNToDiffConv import DenseDCRNNThenDiffConvModel
from models.TimeThenSpace import TimeThenSpaceModel
from models.TemporalSpikeGraphConvNet import TemporalSpikeGraphConvNet
from models.TemporalSynapticGraphConvNet import TemporalSynapticGraphConvNet
from models.TeporalSynapticLeanableWeights import TemporalSynapticLearnableWeights

from models.SynapticGCN import SynapticGCN
from models.MultistepPredictor import MultistepPredictor
from models.SynapticAttention import SynapticAttention
from metrics.weighted_rmse import WeightedRMSE
from metrics.metric_utils import WeatherVariable

from tsl.nn.models import TransformerModel, GraphWaveNetModel
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE
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

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = get_data()
    # train_loader = WeatherDataLoader(data, time=slice('2000', '2002'), temporal_resolution=3)
    # min, max = train_loader.getMinMaxValues()
    # validation_loader = WeatherDataLoader(data, time='2003', temporal_resolution=3, min=min, max=max)
    # min_max = pickle.load(open('./min_max_test.pkl', 'rb'))
    min_max = pickle.load(open('./min_max_train_full.pkl', 'rb'))
    
    # min_max = pickle.load(open('./min_max_global.pkl', 'rb'))
    batch_size = 1
    num_workers = 10
    prefetch_factor = 1
    persistent_workers=False
    pin_memory = False
    train = WeatherDL(
        data,
        time=slice('1992', '2003'),
        temporal_resolution=3,
        min=min_max['min'],
        max=min_max['max'],
        batch_size=batch_size,
        num_workers=num_workers,
        # window=84,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=True
        # multiprocessing_context='fork'
        )
    # min, max = train.data_wrapper.getMinMaxValues()
    valid = WeatherDL(
        data,
        time='2004',
        temporal_resolution=3,
        # horizon=8,
        min=min_max['min'],
        max=min_max['max'],
        batch_size=batch_size,
        num_workers=num_workers,
        # window=84,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        shuffle=True
        # multiprocessing_context='fork'
        )

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
    rnn_layers = 1     #@param
    gnn_kernel = 2     #@param


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
    #     hidden_size=hidden_size // 2
    #     )
    
    # model = TemporalSpikeGraphConvNet(
    #                            input_size=input_size,
    #                            n_nodes=n_nodes,
    #                            horizon=horizon,
    #                            hidden_size=hidden_size * 8,
    #                            use_spike_for_output=True
    #                            )
    output_type = "membrane_potential"
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
    model = TemporalSynapticLearnableWeights(
        input_size=input_size,
        n_nodes=n_nodes,
        horizon=horizon,
        # horizon=8,5
        hidden_size=128,
        output_type=output_type,
        number_of_blocks=3,
        number_of_temporal_steps=3
    )
    
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
    steps_per_day = 24 // 3
    # metrics = {'mse': MaskedMSE(),
    #         #    'mape': MaskedMAPE(),
    #         'mse_at_3_days': MaskedMSE(at=(steps_per_day * 3) - 1),  # 'steps_per_day * 3' indicates the 24th time step,
    #                                         # which correspond to 3 days ahead
    #         'mse_at_5_days': MaskedMSE(at=(steps_per_day * 5) - 1)
    #         }

    variables = [
        WeatherVariable('z', 500),
        WeatherVariable('t', 850)
        ]
    weights = train.data_wrapper.node_weights
    loss_fn = WeightedRMSE(weights, variables=variables)
    metrics = {
        'weighted RMSE': WeightedRMSE(weights, variables=variables),
        'weighted RMSE Z500': WeightedRMSE(weights, variables=[WeatherVariable('z', 500)]),
        'weighted RMSE T850': WeightedRMSE(weights, variables=[WeatherVariable('t', 850)]),
        'weighted RMSE Z500 at day 3': WeightedRMSE(weights, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 3) - 1),
        'weighted RMSE T850 at day 3': WeightedRMSE(weights, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 3) - 1),
        'weighted RMSE Z500 at day 3': WeightedRMSE(weights, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 3) - 1),
        'weighted RMSE T850 at day 3': WeightedRMSE(weights, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 3) - 1),
        'weighted RMSE Z500 at day 5': WeightedRMSE(weights, variables=[WeatherVariable('z', 500)], at=(steps_per_day * 5) - 1),
        'weighted RMSE T850 at day 5': WeightedRMSE(weights, variables=[WeatherVariable('t', 850)], at=(steps_per_day * 5) - 1),
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
        metrics=metrics                # metrics to be logged during train/val/test
    )
    # predictor._set_multistep_attrs(8, horizon)

    print(device,model, torch.cuda.device_count(), model._get_name())


    logger = TensorBoardLogger(save_dir=f"logs/{model._get_name()}", name=f"Weighted RMSE: Full sequence/validation shuffled: {model._get_name()} output=[{output_type}] hidden layer size = [{hidden_size//2}]", version=6,)


    checkpoint_callback = ModelCheckpoint(
        dirpath='logs',
        save_top_k=1,
        monitor='val_weighted RMSE',
        mode='min',
    )
    
    early_stopping = EarlyStopping(monitor='val_weighted RMSE', mode='min', min_delta=1e-5,patience=3, check_on_train_epoch_end=True)
    
    # compiled_predictor = torch.compile(predictor, mode='reduce-overhead')

    trainer = pl.Trainer(max_epochs=2,
                        logger=logger,
                        #  gpus=1 if torch.cuda.is_available() else None,
                        devices=1,
                        accelerator='gpu',
                        #  limit_train_batches=150,
                        # limit_val_batches=50,
                        callbacks=[checkpoint_callback, early_stopping],
                        accumulate_grad_batches=32,
                        val_check_interval=320,
                        limit_val_batches=0.25
                        # num_sanity_val_steps=1,
                        )

    trainer.fit(predictor, train_dataloaders=train.data_loader, val_dataloaders=valid.data_loader)
    

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(seed=15)
    train()