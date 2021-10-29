# Darcy Murphy 2021
from models.ecg_predictor import DomainPredictor, Regressor, Encoder
from utils import Args
import ecg_utils
import ecg_train_utils
import physionet_metrics
from config import constants
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def test_main_and_domain_tasks(encoder, regressor, domain_predictor, dataset, dataset_name=None):
    encoder.eval()
    regressor.eval()
    domain_predictor.eval()
    sigmoid = nn.Sigmoid()
    cuda = torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (data, y_true, domain_true) in enumerate(dataset):
            if cuda:
                data, y_true, domain_true = data.cuda(), y_true.cuda(), domain_true.cuda()

            data, target, domain_target = Variable(data), Variable(y_true), Variable(domain_true)

            features = encoder(data)
            output_pred = regressor(features)
            domain_pred = domain_predictor(features)
            y_pred = sigmoid(output_pred)

            if batch_idx == 0:
                labels_all = y_true
                logits_prob_all = y_pred
                true_domains = domain_true
                pred_domains = domain_pred
            else:
                labels_all = torch.cat((labels_all, y_true), 0)
                logits_prob_all = torch.cat((logits_prob_all, y_pred), 0)
                # this might not be the right way to append for domains as the structure might be different
                true_domains = torch.cat((true_domains, domain_true), 0)
                pred_domains = torch.cat((pred_domains, domain_pred), 0)

    tp_rate = ecg_train_utils.true_positive_rate(labels_all, logits_prob_all)
    challenge_metric = physionet_metrics.calc_accuracy(labels_all, logits_prob_all)
    true_domains = np.argmax(true_domains.detach().cpu().numpy(), axis=1)
    pred_domains = np.argmax(pred_domains.detach().cpu().numpy(), axis=1)
    domain_accuracy = ecg_train_utils.accuracy_score(true_domains, pred_domains)

    print('True positive rate: {:.4f}'.format(tp_rate))
    print('Challenge metric: {:.4f}'.format(challenge_metric))
    print('Domain accuracy: {:.4f}\n'.format(domain_accuracy))

    if dataset_name is not None:
        # use csv file name for linking back to source data if needed
        np.savez(constants.results_filepath_template.format(dataset_name), y_true=labels_all.cpu(), y_pred=logits_prob_all.cpu(),
                 domain_true=true_domains, domain_pred=pred_domains)
    # give accuracy, challenge metric, other scores e.g. sensitivity and specificity
    # want more data on which domain it predicted each example as - break down by class, age, gender etc?


def test_main_task(encoder, regressor, dataset, dataset_name=None):
    encoder.eval()
    regressor.eval()
    sigmoid = nn.Sigmoid()
    cuda = torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (data, y_true, domain_true) in enumerate(dataset):
            if cuda:
                data, y_true, domain_true = data.cuda(), y_true.cuda(), domain_true.cuda()

            data, target, domain_target = Variable(data), Variable(y_true), Variable(domain_true)

            features = encoder(data)
            output_pred = regressor(features)
            y_pred = sigmoid(output_pred)

            if batch_idx == 0:
                labels_all = y_true
                logits_prob_all = y_pred
            else:
                labels_all = torch.cat((labels_all, y_true), 0)
                logits_prob_all = torch.cat((logits_prob_all, y_pred), 0)

    tp_rate = ecg_train_utils.true_positive_rate(labels_all, logits_prob_all)
    challenge_metric = physionet_metrics.calc_accuracy(labels_all, logits_prob_all)

    print('True positive rate: {:.4f}'.format(tp_rate))
    print('Challenge metric: {:.4f}'.format(challenge_metric))

    if dataset_name is not None:
        # use csv file name for linking back to source data if needed
        np.savez(constants.results_filepath_template.format(dataset_name), y_true=labels_all.cpu(),
                 y_pred=logits_prob_all.cpu())
    # give accuracy, challenge metric, other scores e.g. sensitivity and specificity
    # want more data on which domain it predicted each example as - break down by class, age, gender etc?


def test_seen_data(encoder, regressor, domain_predictor, cpsc_test_dataloader, georgia_test_dataloader,
                   ptb_test_dataloader, prefix):
    print("Test CPSC data")
    test_main_and_domain_tasks(encoder, regressor, domain_predictor, cpsc_test_dataloader, prefix + constants.CPSC_TEST)
    print("\nTest Georgia data")
    test_main_and_domain_tasks(encoder, regressor, domain_predictor, georgia_test_dataloader, prefix + constants.GEORGIA_TEST)
    print("\nTest PTB data")
    test_main_and_domain_tasks(encoder, regressor, domain_predictor, ptb_test_dataloader, prefix + constants.PTB_TEST)


if __name__ == "__main__":
    # do this as arg parse at some point
    test_unseen = True
    test_seen = True
    cuda = torch.cuda.is_available()
    if cuda:
        print('Cuda Available')

    args = Args()
    args.batch_size = 18
    args.domain_count = 3

    # load models
    encoder_baseline = Encoder()
    encoder_unlearning = Encoder()
    encoder_unlearning.load_state_dict(torch.load('save_models/full_models/unlearning_epoch31_cm0.5406_encoder.pth'))
    encoder_baseline.load_state_dict(torch.load('save_models/full_models/baseline_epoch42_cm0.5332_encoder.pth'))

    regressor_baseline = Regressor()
    regressor_unlearning = Regressor()
    regressor_unlearning.load_state_dict(torch.load('save_models/full_models/unlearning_epoch31_cm0.5406_regressor.pth'))
    regressor_baseline.load_state_dict(torch.load('save_models/full_models/baseline_epoch42_cm0.5332_regressor.pth'))

    domain_predictor_baseline = DomainPredictor(nodes=3)
    domain_predictor_unlearning = DomainPredictor(nodes=3)
    domain_predictor_unlearning.load_state_dict(torch.load('save_models/full_models/unlearning_epoch31_cm0.5406_domain.pth'))
    domain_predictor_baseline.load_state_dict(torch.load('save_models/full_models/baseline_epoch42_cm0.5332_domain.pth'))

    if cuda:
        encoder_baseline = encoder_baseline.cuda()
        encoder_unlearning = encoder_unlearning.cuda()
        regressor_baseline = regressor_baseline.cuda()
        regressor_unlearning = regressor_unlearning.cuda()
        domain_predictor_baseline = domain_predictor_baseline.cuda()
        domain_predictor_unlearning = domain_predictor_unlearning.cuda()

    if test_seen:
        # load data
        dataloader_batchsize = int(args.batch_size / args.domain_count)

        cpsc_test = ecg_utils.load_test_data(constants.data_split_template.format(constants.CPSC_TEST), args.domain_count, 0)
        cpsc_test_dataloader = DataLoader(cpsc_test, batch_size=dataloader_batchsize, shuffle=False, num_workers=0)
        georgia_test = ecg_utils.load_test_data(constants.data_split_template.format(constants.GEORGIA_TEST), args.domain_count, 1)
        georgia_test_dataloader = DataLoader(georgia_test, batch_size=dataloader_batchsize, shuffle=False, num_workers=0)
        ptb_test = ecg_utils.load_test_data(constants.data_split_template.format(constants.PTB_TEST), args.domain_count, 2)
        ptb_test_dataloader = DataLoader(ptb_test, batch_size=dataloader_batchsize, shuffle=False, num_workers=0)

        print('Seen data, baseline model')
        test_seen_data(encoder_baseline, regressor_baseline, domain_predictor_baseline,
                       cpsc_test_dataloader, georgia_test_dataloader, ptb_test_dataloader, 'baseline_')
        print('\nSeen data, unlearning model')
        test_seen_data(encoder_unlearning, regressor_unlearning, domain_predictor_unlearning,
                       cpsc_test_dataloader, georgia_test_dataloader, ptb_test_dataloader, 'unlearning_')

    if test_unseen:
        shaoxing_test = ecg_utils.load_test_data('data_splits/shaoxing.csv', 1, 0)
        shaoxing_test_dataloader = DataLoader(shaoxing_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print('\nShaoxing data, baseline model')
        test_main_task(encoder_baseline, regressor_baseline, shaoxing_test_dataloader, 'shaoxing_baseline')
        print('\nShaoxing data, unlearning model')
        test_main_task(encoder_unlearning, regressor_unlearning, shaoxing_test_dataloader, 'shaoxing_unlearning')
