from utils import Args
from torch.utils.data import DataLoader
import ecg_utils
from models.ecg_predictor import Regressor, Encoder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ecg_train_utils_no_unlearning import train_only, val_only
import sys


LOSS_PATH = 'loss_pth'
PRE_TRAIN_ENCODER = 'pretrain_encoder'
PRE_TRAIN_REGRESSOR = 'pretrain_regressor'
PRE_TRAIN_DOMAIN = 'pretrain_domain'

PATH_ENCODER = 'encoder_pth'
PATH_REGRESSOR = 'regressor_pth'
PATH_DOMAIN = 'domain_pth'

CHK_PATH_ENCODER = 'encoder_chk_pth'
CHK_PATH_REGRESSOR = 'regressor_chk_pth'
CHK_PATH_DOMAIN = 'domain_chk_pth'


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        print('Cuda Available', flush=True)

    args = Args()
    args.batch_size = 3
    args.domain_count = 3
    args.learning_rate = 1e-4
    args.patience = 50
    args.epochs = 20
    args.epoch_reached = 1
    args.epoch_stage_1 = 25
    args.log_interval = 100

    # load data
    clean_train_path = 'data_splits/cpsc_clean_train.csv'
    clean_val_path = 'data_splits/cpsc_clean_val.csv'
    clean_train_dataset, clean_val_dataset = ecg_utils.load_data(clean_train_path, clean_val_path, args.domain_count, 0)

    gaus_train_path = 'data_splits/cpsc_gaus_train.csv'
    gaus_val_path = 'data_splits/cpsc_gaus_val.csv'
    gaus_train_dataset, gaus_val_dataset = ecg_utils.load_data(gaus_train_path, gaus_val_path, args.domain_count, 1)

    sin_train_path = 'data_splits/cpsc_sin_train.csv'
    sin_val_path = 'data_splits/cpsc_sin_val.csv'
    sin_train_dataset, sin_val_dataset = ecg_utils.load_data(sin_train_path, sin_val_path, args.domain_count, 2)

    c_train_dataloader = DataLoader(clean_train_dataset, batch_size=1, shuffle=True, num_workers=0)
    c_val_dataloader = DataLoader(clean_val_dataset, batch_size=1, shuffle=True, num_workers=0)
    g_train_dataloader = DataLoader(gaus_train_dataset, batch_size=1, shuffle=True, num_workers=0)
    g_val_dataloader = DataLoader(gaus_val_dataset, batch_size=1, shuffle=True, num_workers=0)
    s_train_dataloader = DataLoader(sin_train_dataset, batch_size=1, shuffle=True, num_workers=0)
    s_val_dataloader = DataLoader(sin_val_dataset, batch_size=1, shuffle=True, num_workers=0)

    # Load the model
    encoder = Encoder()
    regressor = Regressor()

    if cuda:
        encoder = encoder.cuda()
        regressor = regressor.cuda()

    criteron = nn.MSELoss()

    if cuda:
        criteron = criteron.cuda()

    optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=args.learning_rate)

    models = [encoder, regressor]
    optimizers = [optimizer]
    train_dataloaders = [c_train_dataloader, g_train_dataloader, s_train_dataloader]
    val_dataloaders = [c_val_dataloader, g_val_dataloader, s_val_dataloader]
    criterions = [criteron]

    for epoch in range(args.epoch_reached, args.epochs + 1):
        if epoch < args.epochs:
            print('Training Main Encoder')
            print('Epoch ', epoch, '/', args.epochs, flush=True)
            loss = train_only(args, models, train_dataloaders,
                              optimizers, criterions, epoch)
            torch.cuda.empty_cache()  # Clear memory cache
            val_loss = val_only(args, models, val_dataloaders, criterions)

