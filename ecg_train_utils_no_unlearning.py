import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import physionet_metrics
import ecg_train_utils
import torch.nn as nn


def train_only(args, models, train_loaders, optimizers, criterions, epoch):
    [encoder, regressor] = models
    [optimizer] = optimizers
    [b_train_dataloader, o_train_dataloader, w_train_dataloader] = train_loaders
    [criteron] = criterions

    regressor_loss = 0
    sigmoid = nn.Sigmoid()
    encoder.train()
    regressor.train()

    batches = 0
    for batch_idx, (b, o, w) in enumerate(zip(b_train_dataloader, o_train_dataloader, w_train_dataloader)):

        all_data = ecg_train_utils.get_batch_split(b, o, w, args.batch_size)
        if True:
            # (n1, n2, n3) = splits
            (data, target, domain_target) = all_data

            if list(data.size())[0] == args.batch_size:
                batches += 1

                # First update the encoder and regressor
                optimizer.zero_grad()
                features = encoder(data)
                output_pred = regressor(features)
                y_pred = sigmoid(output_pred)
                if batch_idx == 0:
                    labels_all = target
                    logits_prob_all = y_pred
                else:
                    labels_all = torch.cat((labels_all, target), 0)
                    logits_prob_all = torch.cat((logits_prob_all, y_pred), 0)
                r_loss = ecg_train_utils.calculate_regression_loss(criteron, y_pred, target)
                # r_loss = criteron(y_pred, target)
                regressor_loss += r_loss * data.size(0)
                r_loss.backward()
                optimizer.step()



                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Regressor Loss: {:.6f}'.format(
                        epoch, (batch_idx+1) * len(data), len(b_train_dataloader.dataset),
                               100. * (batch_idx+1) / len(b_train_dataloader), r_loss.item()), flush=True)
                    tp_rate = ecg_train_utils.true_positive_rate(labels_all, logits_prob_all)
                    challenge_metric = physionet_metrics.calc_accuracy(labels_all, logits_prob_all, '.', threshold=0.5)
                    print('Regressor Loss: {:.4f}'.format(r_loss, flush=True))
                    print('True positive rate: {:.4f}'.format(tp_rate), flush=True)
                    print('Challenge Metric: {:.4f}'.format(challenge_metric, flush=True))

                # del target
                # del r_loss
                # del features

    av_loss = regressor_loss / batches
    tp_rate = ecg_train_utils.true_positive_rate(labels_all, logits_prob_all)
    challenge_metric = physionet_metrics.calc_accuracy(labels_all, logits_prob_all, '.', threshold=0.5)
    print('\nTraining set: Average loss: {:.4f}'.format(av_loss, flush=True))
    print('True positive rate: {:.4f}'.format(tp_rate), flush=True)
    print('Challenge Metric: {:.4f}'.format(challenge_metric, flush=True))

    return av_loss


def val_only(args, models, val_loaders, criterions):
    [encoder, regressor] = models
    [b_val_dataloader, o_val_dataloader, w_val_dataloader] = val_loaders
    [criteron] = criterions
    sigmoid = nn.Sigmoid()
    encoder.eval()
    regressor.eval()

    regressor_loss = 0

    batches = 0
    with torch.no_grad():
        for batch_idx, (b, o, w) in enumerate(zip(b_val_dataloader, o_val_dataloader, w_val_dataloader)):

            all_data = ecg_train_utils.get_batch_split(b, o, w, args.batch_size)
            if True:
                (data, target, domain_target) = all_data

                if list(data.size())[0] == args.batch_size:
                    batches += 1
                    features = encoder(data)
                    output_pred = regressor(features)
                    y_pred = sigmoid(output_pred)

                    if batch_idx == 0:
                        labels_all = target
                        logits_prob_all = y_pred
                    else:
                        labels_all = torch.cat((labels_all, target), 0)
                        logits_prob_all = torch.cat((logits_prob_all, y_pred), 0)

                    r_loss = ecg_train_utils.calculate_regression_loss(criteron, y_pred, target)
                    regressor_loss += r_loss

    val_loss = regressor_loss / batches
    tp_rate = ecg_train_utils.true_positive_rate(labels_all, logits_prob_all)
    challenge_metric = physionet_metrics.calc_accuracy(labels_all, logits_prob_all, '.', threshold=0.5)

    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss, flush=True))
    print('True positive rate: {:.4f}'.format(tp_rate), flush=True)
    print('Challenge Metric: {:.4f}'.format(challenge_metric, flush=True))

    return val_loss
