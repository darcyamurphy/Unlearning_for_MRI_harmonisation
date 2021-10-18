from utils import Args, EarlyStopping_unlearning
from torch.utils.data import DataLoader
import ecg_utils
from models.ecg_predictor import DomainPredictor, Regressor, Encoder
import torch
import torch.nn as nn
from losses.confusion_loss import confusion_loss
import torch.optim as optim
import numpy as np
from ecg_train_utils import train_unlearn_threedatasets, val_unlearn_threedatasets, train_encoder_unlearn_threedatasets
import sys
from config.constants import *


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        print('Cuda Available', flush=True)

    args = Args()
    args.batch_size = 18
    args.domain_count = 3
    args.learning_rate = 1e-4
    args.patience = 50
    args.epochs = 3
    args.epoch_reached = 1
    args.epoch_stage_1 = 2
    args.log_interval = 100
    args.alpha = 1
    args.beta = 10

    # load data
    train_dataset_a, val_dataset_a = ecg_utils.load_data(DOMAIN_A_TRAIN_PATH, DOMAIN_A_VAL_PATH, args.domain_count, 0)
    train_dataset_b, val_dataset_b = ecg_utils.load_data(DOMAIN_B_TRAIN_PATH, DOMAIN_B_VAL_PATH, args.domain_count, 1)
    train_dataset_c, val_dataset_c = ecg_utils.load_data(DOMAIN_C_TRAIN_PATH, DOMAIN_C_VAL_PATH, args.domain_count, 2)

    dataloader_batchsize = int(args.batch_size/args.domain_count)
    c_train_dataloader = DataLoader(train_dataset_a, batch_size=dataloader_batchsize, shuffle=True, num_workers=0)
    c_val_dataloader = DataLoader(val_dataset_a, batch_size=dataloader_batchsize, shuffle=True, num_workers=0)
    g_train_dataloader = DataLoader(train_dataset_b, batch_size=dataloader_batchsize, shuffle=True, num_workers=0)
    g_val_dataloader = DataLoader(val_dataset_b, batch_size=dataloader_batchsize, shuffle=True, num_workers=0)
    s_train_dataloader = DataLoader(train_dataset_c, batch_size=dataloader_batchsize, shuffle=True, num_workers=0)
    s_val_dataloader = DataLoader(val_dataset_c, batch_size=dataloader_batchsize, shuffle=True, num_workers=0)

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

    optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-4)
    optimizer_conf = optim.Adam(list(encoder.parameters()), lr=1e-4)
    optimizer_dm = optim.Adam(list(domain_predictor.parameters()), lr=1e-7)

    # Initalise the early stopping
    early_stopping = EarlyStopping_unlearning(args.patience, verbose=False)

    loss_store = []

    models = [encoder, regressor, domain_predictor]
    optimizers = [optimizer, optimizer_conf, optimizer_dm]
    train_dataloaders = [c_train_dataloader, g_train_dataloader, s_train_dataloader]
    val_dataloaders = [c_val_dataloader, g_val_dataloader, s_val_dataloader]
    criterions = [criteron, conf_criterion, domain_criterion]

    for epoch in range(args.epoch_reached, args.epochs + 1):
        if epoch < args.epoch_stage_1:
            print('Training Main Encoder')
            print('Epoch ', epoch, '/', args.epochs, flush=True)
            loss, acc, dm_loss, conf_loss = train_encoder_unlearn_threedatasets(args, models, train_dataloaders,
                                                                                optimizers, criterions, epoch)
            torch.cuda.empty_cache()  # Clear memory cache
            val_loss, val_acc = val_unlearn_threedatasets(args, models, val_dataloaders, criterions)
            loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])

            np.save(LOSS_PATH, np.array(loss_store))

            if epoch == args.epoch_stage_1 - 1:
                torch.save(encoder.state_dict(), PRE_TRAIN_ENCODER)
                torch.save(regressor.state_dict(), PRE_TRAIN_REGRESSOR)
                torch.save(domain_predictor.state_dict(), PRE_TRAIN_DOMAIN)

        else:
            print('Unlearning')
            print('Epoch ', epoch, '/', args.epochs, flush=True)

            if epoch == args.epoch_stage_1:
                print("Validation before unlearning:")
                val_loss, val_acc = val_unlearn_threedatasets(args, models, val_dataloaders, criterions)

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