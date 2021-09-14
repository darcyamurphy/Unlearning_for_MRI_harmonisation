from utils import Args, EarlyStopping_unlearning
from torch.utils.data import DataLoader
import ecg_utils
from models.ecg_predictor import DomainPredictor, Regressor, Encoder
import torch
import torch.nn as nn
from losses.confusion_loss import confusion_loss
import torch.optim as optim
import numpy as np
from train_utils import train_unlearn_threedatasets, val_unlearn_threedatasets, train_encoder_unlearn_threedatasets, val_encoder_unlearn_threedatasets
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
    args.epochs = 300
    args.batch_size = 1
    args.domain_count = 3
    args.learning_rate = 1e-4
    args.patience = 50
    args.epochs = 300
    args.epoch_reached = 1
    args.epoch_stage_1 = 100

    # load data
    clean_train_path = 'clean_train.csv'
    clean_val_path = 'clean_val.csv'
    clean_train_dataset, clean_val_dataset = ecg_utils.load_data(clean_train_path, clean_val_path)

    gaus_train_path = 'gaus_train.csv'
    gaus_val_path = 'gaus_val.csv'
    gaus_train_dataset, gaus_val_dataset = ecg_utils.load_data(gaus_train_path, gaus_val_path)

    sin_train_path = 'sin_train.csv'
    sin_val_path = 'sin_val.csv'
    sin_train_dataset, sin_val_dataset = ecg_utils.load_data(sin_train_path, sin_val_path)

    c_train_dataloader = DataLoader(clean_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    c_val_dataloader = DataLoader(clean_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    g_train_dataloader = DataLoader(gaus_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    g_val_dataloader = DataLoader(gaus_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    s_train_dataloader = DataLoader(sin_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    s_val_dataloader = DataLoader(sin_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Load the model
    encoder = Encoder()
    regressor = Regressor()
    domain_predictor = DomainPredictor(nodes=args.domain_count)

    if cuda:
        encoder = encoder.cuda()
        regressor = regressor.cuda()
        domain_predictor = domain_predictor.cuda()

    criteron = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()
    conf_criterion = confusion_loss()

    if cuda:
        criteron = criteron.cuda()
        domain_criterion = domain_criterion.cuda()
        conf_criterion = conf_criterion.cuda()

    optimizer_step1 = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()) +
                                 list(domain_predictor.parameters()), lr=args.learning_rate)

    optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-6)
    optimizer_conf = optim.Adam(list(encoder.parameters()), lr=1e-6)
    optimizer_dm = optim.Adam(list(domain_predictor.parameters()), lr=1e-6)

    optimizers = [optimizer, optimizer_conf, optimizer_dm]

    # Initalise the early stopping
    early_stopping = EarlyStopping_unlearning(args.patience, verbose=False)

    loss_store = []

    models = [encoder, regressor, domain_predictor]
    train_dataloaders = [c_train_dataloader, g_train_dataloader, s_train_dataloader]
    val_dataloaders = [c_val_dataloader, g_val_dataloader, s_val_dataloader]
    criterions = [criteron, conf_criterion, domain_criterion]

    for epoch in range(args.epoch_reached, args.epochs + 1):
        if epoch < args.epoch_stage_1:
            print('Training Main Encoder')
            print('Epoch ', epoch, '/', args.epochs, flush=True)

            loss, acc, dm_loss, conf_loss = train_encoder_unlearn_threedatasets(args, models, train_dataloaders,
                                                                                [optimizer_step1], criterions, epoch)
            torch.cuda.empty_cache()  # Clear memory cache
            val_loss, val_acc = val_encoder_unlearn_threedatasets(args, models, val_dataloaders, criterions)
            loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])

            np.save(LOSS_PATH, np.array(loss_store))

            if epoch == args.epoch_stage_1 - 1:
                torch.save(encoder.state_dict(), PRE_TRAIN_ENCODER)
                torch.save(regressor.state_dict(), PRE_TRAIN_REGRESSOR)
                torch.save(domain_predictor.state_dict(), PRE_TRAIN_DOMAIN)

        else:
            print('Unlearning')
            print('Epoch ', epoch, '/', args.epochs, flush=True)

            loss, acc, dm_loss, conf_loss = train_unlearn_threedatasets(args, models, train_dataloaders, optimizers,
                                                                        criterions, epoch)
            torch.cuda.empty_cache()  # Clear memory cache
            val_loss, val_acc = val_unlearn_threedatasets(args, models, val_dataloaders, criterions)

            loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])
            np.save(LOSS_PATH, np.array(loss_store))

            # Decide whether the model should stop training or not
            early_stopping(val_loss, models, epoch, optimizer, loss,
                           [CHK_PATH_ENCODER, CHK_PATH_REGRESSOR, CHK_PATH_DOMAIN])
            if early_stopping.early_stop:
                loss_store = np.array(loss_store)
                np.save(LOSS_PATH, loss_store)
                sys.exit('Patience Reached - Early Stopping Activated')

            if epoch == args.epochs:
                print('Finished Training', flush=True)
                print('Saving the model', flush=True)

                # Save the model in such a way that we can continue training later
                torch.save(encoder.state_dict(), PATH_ENCODER)
                torch.save(regressor.state_dict(), PATH_REGRESSOR)
                torch.save(domain_predictor.state_dict(), PATH_DOMAIN)

                loss_store = np.array(loss_store)
                np.save(LOSS_PATH, loss_store)

            torch.cuda.empty_cache()  # Clear memory cache