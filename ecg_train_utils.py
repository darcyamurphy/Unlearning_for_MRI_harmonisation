import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast
import physionet_metrics
import torch.nn as nn


def combine_datasets(b, o, w):
    (b_data, b_target, b_domain) = b
    (o_data, o_target, o_domain) = o
    (w_data, w_target, w_domain) = w

    data = torch.cat((b_data, o_data, w_data), 0)
    target = torch.cat((b_target, o_target, w_target), 0)
    domain_target = torch.cat((b_domain, o_domain, w_domain), 0)

    cuda = torch.cuda.is_available()
    if cuda:
        data, target, domain_target = data.cuda(), target.cuda(), domain_target.cuda()

    data, target, domain_target = Variable(data), Variable(target), Variable(domain_target)

    return (data, target, domain_target)


def true_positive_rate(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    true_positives = 0
    false_negatives = 0
    for i in range(0, len(y_true)):
        for j in range(0, len(y_true[0])):
            if y_pred[i][j] >= 0.5 and y_true[i][j] == 1:
                true_positives += 1
            elif y_pred[i][j] < 0.5 and y_true[i][j] == 1:
                false_negatives += 1

    if true_positives + false_negatives == 0:
        return 0
    return true_positives/(true_positives + false_negatives)


def print_metrics(domains_true, domains_pred, y_true, y_pred, avg_domain_loss, avg_regressor_loss):
    acc = accuracy_score(domains_true, domains_pred)
    tp_rate = true_positive_rate(y_true, y_pred)
    challenge_metric = physionet_metrics.calc_accuracy(y_true, y_pred)

    print('Average Regressor loss: {:.4f}'.format(avg_regressor_loss))
    print('Average Domain loss: {:.4f}'.format(avg_domain_loss))
    print('Average Domain Acc: {:.4f}'.format(acc))
    print('True positive rate: {:.4f}'.format(tp_rate))
    print('Challenge Metric: {:.4f}\n'.format(challenge_metric))


def train_encoder_unlearn_threedatasets(args, models, train_loaders, optimizers, criterions, epoch):
    [encoder, regressor, domain_predictor] = models
    [optimizer, _, optimizer_dm] = optimizers
    [b_train_dataloader, o_train_dataloader, w_train_dataloader] = train_loaders
    [criteron, _, domain_criterion] = criterions

    regressor_loss = 0
    domain_loss = 0

    sigmoid = nn.Sigmoid()
    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    batches = 0
    for batch_idx, (b, o, w) in enumerate(zip(b_train_dataloader, o_train_dataloader, w_train_dataloader)):

        (data, target, domain_target) = combine_datasets(b, o, w)


        if list(data.size())[0] == args.batch_size :
            batches += 1

            # First update the encoder and regressor
            optimizer.zero_grad()
            features = encoder(data)
            output_pred = regressor(features)
            y_pred = sigmoid(output_pred)
            r_loss = criteron(y_pred, target)

            if batch_idx == 0:
                labels_all = target
                logits_prob_all = y_pred
            else:
                labels_all = torch.cat((labels_all, target), 0)
                logits_prob_all = torch.cat((logits_prob_all, y_pred), 0)

            regressor_loss += r_loss * data.size(0)
            r_loss.backward()
            optimizer.step()

            domain_pred = domain_predictor(features.detach())
            d_loss = domain_criterion(domain_pred, torch.max(domain_target, 1)[1])
            d_loss.backward()
            optimizer_dm.step()

            regressor_loss += r_loss
            domain_loss += d_loss

            domains = np.argmax(domain_pred.detach().cpu().numpy(), axis=1)
            domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
            true_domains.append(domain_target)
            pred_domains.append(domains)


            if batch_idx % args.log_interval == 0:
                tp_rate = true_positive_rate(labels_all, logits_prob_all)
                challenge_metric = physionet_metrics.calc_accuracy(labels_all, logits_prob_all)
                total_length = len(b_train_dataloader.dataset)*3
                print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, (batch_idx+1) * len(data), total_length,
                           100. * (batch_idx+1) / len(b_train_dataloader.dataset)))
                print('\t Regressor Loss: {:.4f}'.format(r_loss))
                print('\t Domain Loss: {:.4f}'.format(d_loss))
                print('\t True positive rate: {:.4f}'.format(tp_rate))
                print('\t Challenge Metric: {:.4f}'.format(challenge_metric))

            del target
            del r_loss
            del d_loss
            del features

    av_loss = regressor_loss / batches
    av_dom_loss = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    domain_acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set: ')
    print_metrics(true_domains, pred_domains, labels_all, logits_prob_all, av_dom_loss, av_loss)

    return av_loss, domain_acc, av_dom_loss, np.NaN


def train_unlearn_threedatasets(args, models, train_loaders, optimizers, criterions, epoch):
    [encoder, regressor, domain_predictor] = models
    [optimizer, optimizer_conf, optimizer_dm] = optimizers
    [b_train_dataloader, o_train_dataloader, w_train_dataloader] = train_loaders
    [criteron, conf_criterion, domain_criterion] = criterions

    regressor_loss = 0
    domain_loss = 0
    conf_loss = 0

    sigmoid = nn.Sigmoid()
    encoder.train()
    regressor.train()
    domain_predictor.train()

    true_domains = []
    pred_domains = []

    first_batch = True
    torch.autograd.set_detect_anomaly(True)

    batches = 0
    for batch_idx, (b, o, w) in enumerate(zip(b_train_dataloader, o_train_dataloader, w_train_dataloader)):
        (data, target, domain_target) = combine_datasets(b, o, w)

        if list(data.size())[0] == args.batch_size :
            batches += 1

            # First update the encoder and regressor
            optimizer.zero_grad()
            features = encoder(data)
            output_pred = regressor(features)
            y_pred = sigmoid(output_pred)

            if first_batch:
                first_batch = False
                labels_all = target
                logits_prob_all = y_pred
            else:
                labels_all = torch.cat((labels_all, target), 0)
                logits_prob_all = torch.cat((logits_prob_all, y_pred), 0)

            loss = criteron(y_pred, target)
            loss.backward()
            optimizer.step()

            # Now update just the domain classifier
            optimizer_dm.zero_grad()
            output_dm = domain_predictor(features.detach())
            loss_dm = args.alpha * domain_criterion(output_dm, torch.max(domain_target, 1)[1])
            loss_dm.backward()
            optimizer_dm.step()

            # something is going horribly wrong here
            # Now update just the encoder using the domain loss
            optimizer_conf.zero_grad()
            with autocast():
                features = encoder(data)
                output_dm_conf = domain_predictor(features)
                loss_conf = args.beta * conf_criterion(output_dm_conf, domain_target)
            loss_conf.backward()
            optimizer_conf.step()

            regressor_loss += loss
            domain_loss += loss_dm
            conf_loss += loss_conf

            output_dm_conf = np.argmax(output_dm_conf.detach().cpu().numpy(), axis=1)
            domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)
            true_domains.append(domain_target)
            pred_domains.append(output_dm_conf)

            if batch_idx % args.log_interval == 0:
                total_length = len(b_train_dataloader.dataset)*3
                print('Train Unlearning Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, (batch_idx+1) * len(data), total_length,
                           100. * (batch_idx+1) / len(b_train_dataloader.dataset)))
                tp_rate = true_positive_rate(labels_all, logits_prob_all)
                challenge_metric = physionet_metrics.calc_accuracy(labels_all, logits_prob_all)
                print('\t Confusion loss = ', loss_conf.item())
                print('\t Domain Loss = ', loss_dm.item())
                print('\t Regressor Loss: {:.4f}'.format(regressor_loss/batches))
                print('\t True positive rate: {:.4f}'.format(tp_rate))
                print('\t Challenge Metric: {:.4f}'.format(challenge_metric))

            del target
            del loss
            del features
            torch.cuda.empty_cache()

    av_loss = regressor_loss / batches
    av_conf = conf_loss / batches
    av_dom = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('\nTraining set:')
    print('Average Conf loss: {:.4f}'.format(av_conf))
    print_metrics(true_domains, pred_domains, labels_all, logits_prob_all, av_dom, av_loss)

    return av_loss, acc, av_dom, av_conf


def val_unlearn_threedatasets(args, models, val_loaders, criterions):
    [encoder, regressor, domain_predictor] = models
    [b_val_dataloader, o_val_dataloader, w_val_dataloader] = val_loaders
    [criteron, _, domain_criterion] = criterions

    encoder.eval()
    regressor.eval()
    domain_predictor.eval()
    sigmoid = nn.Sigmoid()

    regressor_loss = 0
    domain_loss = 0

    true_domains = []
    pred_domains = []

    batches = 0
    with torch.no_grad():
        for batch_idx, (b, o, w) in enumerate(zip(b_val_dataloader, o_val_dataloader, w_val_dataloader)):

            (data, target, domain_target) = combine_datasets(b, o, w)

            if list(data.size())[0] == args.batch_size:
                batches += 1
                features = encoder(data)
                output_pred = regressor(features)
                y_pred = sigmoid(output_pred)

                if batches == 1:
                    labels_all = target
                    logits_prob_all = y_pred
                else:
                    labels_all = torch.cat((labels_all, target), 0)
                    logits_prob_all = torch.cat((logits_prob_all, y_pred), 0)

                loss = criteron(y_pred, target)
                regressor_loss += loss

                domain_pred = domain_predictor.forward(features)
                d_loss = domain_criterion(domain_pred, torch.max(domain_target, 1)[1])
                domain_pred = np.argmax(domain_pred.detach().cpu().numpy(), axis=1)
                domain_loss += d_loss

                # sometimes domain target is a tensor and sometimes it's already an ndarray and i don't know why!
                if not isinstance(domain_target, np.ndarray):
                    domain_target = np.argmax(domain_target.detach().cpu().numpy(), axis=1)

                true_domains.append(domain_target)
                pred_domains.append(domain_pred)

    avg_regressor_loss = regressor_loss / batches
    avg_domain_loss = domain_loss / batches

    true_domains = np.array(true_domains).reshape(-1)
    pred_domains = np.array(pred_domains).reshape(-1)
    acc = accuracy_score(true_domains, pred_domains)

    print('Validation set: ')
    print_metrics(true_domains, pred_domains, labels_all, logits_prob_all, avg_domain_loss, avg_regressor_loss)

    return avg_regressor_loss, acc

