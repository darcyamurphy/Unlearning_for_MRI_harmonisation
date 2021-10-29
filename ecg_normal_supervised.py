# Darcy Murphy 2021
# Adapted from code by Nicola Dinsdale 2020
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


def save_all_models(prefix, encoder, regressor, domain_predictor):
    torch.save(encoder.state_dict(), PATH_ENCODER.format(prefix))
    torch.save(regressor.state_dict(), PATH_REGRESSOR.format(prefix))
    torch.save(domain_predictor.state_dict(), PATH_DOMAIN.format(prefix))


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        print('Cuda Available', flush=True)

    args = Args()
    args.batch_size = 18
    args.learning_rate = 1e-4
    args.patience = 25
    args.epochs = 70
    args.begin_epoch = 1
    args.epoch_stage_1 = 20
    args.log_interval = 100
    args.alpha = 1
    args.beta = 10
    args.save_frequency = 3
    args.resume = True
    args.unlearning = True

    # load data
    train_datasets, val_datasets = ecg_utils.load_unbalanced_data([DOMAIN_A_TRAIN_PATH, DOMAIN_B_TRAIN_PATH, DOMAIN_C_TRAIN_PATH],
                                                                  [DOMAIN_A_VAL_PATH, DOMAIN_B_VAL_PATH, DOMAIN_C_VAL_PATH])

    train_dataloaders = []
    val_dataloaders = []
    dataloader_batchsize = int(args.batch_size/len(train_datasets))
    for i in range(len(train_datasets)):
        train_dataloader = DataLoader(train_datasets[i], batch_size=dataloader_batchsize, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_datasets[i], batch_size=dataloader_batchsize, shuffle=True, num_workers=0)
        train_dataloaders.append(train_dataloader)
        val_dataloaders.append(val_dataloader)

    # Load the model
    encoder = Encoder()
    regressor = Regressor()
    domain_predictor = DomainPredictor(nodes=len(train_datasets))

    if cuda:
        encoder = encoder.cuda()
        regressor = regressor.cuda()
        domain_predictor = domain_predictor.cuda()

    if args.resume:
        encoder.load_state_dict(torch.load('save_models/epoch17_pretrain_cm0.5328_encoder.pth'))
        regressor.load_state_dict(torch.load('save_models/epoch17_pretrain_cm0.5328_regressor.pth'))
        domain_predictor.load_state_dict(torch.load('save_models/epoch17_pretrain_cm0.5328_domain.pth'))
        args.begin_epoch = 18
        print('resuming from {}'.format(args.begin_epoch))

    criteron = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()
    conf_criterion = confusion_loss()

    if cuda:
        criteron = criteron.cuda()
        domain_criterion = domain_criterion.cuda()
        conf_criterion = conf_criterion.cuda()

    optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-4)
    optimizer_conf = optim.Adam(list(encoder.parameters()), lr=1e-6)
    optimizer_dm = optim.Adam(list(domain_predictor.parameters()), lr=1e-7)

    # Initalise the early stopping
    early_stopping = EarlyStopping_unlearning(args.patience, verbose=False)

    loss_store = []

    models = [encoder, regressor, domain_predictor]
    optimizers = [optimizer, optimizer_conf, optimizer_dm]
    criterions = [criteron, conf_criterion, domain_criterion]

    best_cm = 0
    for epoch in range(args.begin_epoch, args.epochs + 1):
        if epoch < args.epoch_stage_1:
            print('Training Main Encoder')
            print('Epoch ', epoch, '/', args.epochs)
            loss, acc, dm_loss, conf_loss = train_encoder_unlearn_threedatasets(args, models, train_dataloaders,
                                                                                optimizers, criterions, epoch)
            torch.cuda.empty_cache()  # Clear memory cache
            val_loss, val_acc, cm = val_unlearn_threedatasets(args, models, val_dataloaders, criterions)
            loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])

            np.save(LOSS_PATH, np.array(loss_store))

            if epoch == args.epoch_stage_1 - 1:
                save_all_models('final_pretrain_epoch{}_cm{:.4f}_'.format(epoch, cm), encoder, regressor, domain_predictor)
            elif cm > best_cm:
                print("Better val challange metric score.\n"
                      "Previous: {:.4f} \t New: {:.4f}"
                      "Saving models".format(best_cm, cm))
                best_cm = cm
                save_all_models('epoch{}_pretrain_cm{:.4f}_'.format(epoch, cm), encoder, regressor, domain_predictor)
            elif epoch % args.save_frequency == 0:
                print('Checkpoint, saving model.')
                save_all_models('checkpoint_epoch{}_pretrain_cm{:.4f}_'.format(epoch, cm), encoder, regressor,
                                domain_predictor)
        else:
            if epoch == args.epoch_stage_1:
                print("Validation before stage 2:")
                val_loss, val_acc, cm = val_unlearn_threedatasets(args, models, val_dataloaders, criterions)
                # lower learning rate on main task for unlearning stage
                optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=1e-6)
                optimizers = [optimizer, optimizer_conf, optimizer_dm]

            if args.unlearning:
                print('Unlearning')
                print('Epoch ', epoch, '/', args.epochs)

                loss, acc, dm_loss, conf_loss = train_unlearn_threedatasets(args, models, train_dataloaders, optimizers,
                                                                            criterions, epoch)
                torch.cuda.empty_cache()  # Clear memory cache
                val_loss, val_acc, cm = val_unlearn_threedatasets(args, models, val_dataloaders, criterions)

                loss_store.append([loss, val_loss, acc, val_acc, dm_loss, conf_loss])
                np.save(LOSS_PATH, np.array(loss_store))
            else:
                print("Lowered learning rate")
                print('Epoch ', epoch, '/', args.epochs)
                loss, acc, dm_loss, conf_loss = train_encoder_unlearn_threedatasets(args, models, train_dataloaders,
                                                                                    optimizers, criterions, epoch)
                torch.cuda.empty_cache()  # Clear memory cache
                val_loss, val_acc, cm = val_unlearn_threedatasets(args, models, val_dataloaders, criterions)
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
                save_all_models('final_model_epoch{}_cm{:.4f}_'.format(epoch, cm), encoder, regressor, domain_predictor)
                loss_store = np.array(loss_store)
                np.save(LOSS_PATH, loss_store)
            elif cm > best_cm:
                print("Better val challange metric score.\n"
                      "Previous: {:.4f} \t New: {:.4f}"
                      "Saving models".format(best_cm, cm))
                best_cm = cm
                if args.unlearning:
                    save_all_models('unlearning_epoch{}_cm{:.4f}_'.format(epoch, cm), encoder, regressor, domain_predictor)
                else:
                    save_all_models('baseline_epoch{}_cm{:.4f}_'.format(epoch, cm), encoder, regressor, domain_predictor)
            elif epoch % args.save_frequency == 0:
                print("Checkpoint, saving model")
                if args.unlearning:
                    save_all_models('checkpoint_epoch{}_unlearning_cm{:.4f}_'.format(epoch, cm), encoder, regressor, domain_predictor)
                else:
                    save_all_models('checkpoint_epoch{}_baseline_cm{:.4f}_'.format(epoch, cm), encoder, regressor,
                                    domain_predictor)



        torch.cuda.empty_cache()  # Clear memory cache