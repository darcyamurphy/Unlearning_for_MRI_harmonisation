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
        np.savez(constants.results_filepath_template.format(dataset_name), y_true=labels_all, y_pred=logits_prob_all,
                 domain_true=true_domains, domain_pred=pred_domains)
    # give accuracy, challenge metric, other scores e.g. sensitivity and specificity
    # want more data on which domain it predicted each example as - break down by class, age, gender etc?


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        print('Cuda Available')

    args = Args()
    args.batch_size = 18
    args.domain_count = 3

    # load models
    encoder = Encoder()
    encoder.load_state_dict(torch.load(constants.PATH_ENCODER))

    regressor = Regressor()
    regressor.load_state_dict(torch.load(constants.PATH_REGRESSOR))

    domain_predictor = DomainPredictor(nodes=3)
    domain_predictor.load_state_dict(torch.load(constants.PATH_DOMAIN))

    if cuda:
        encoder = encoder.cuda()
        regressor = regressor.cuda()
        domain_predictor = domain_predictor.cuda()

    # load data
    dataloader_batchsize = int(args.batch_size / args.domain_count)

    cpsc_test = ecg_utils.load_test_data(constants.CPSC_TEST, args.domain_count, 0)
    cpsc_test_dataloader = DataLoader(cpsc_test, batch_size=dataloader_batchsize, shuffle=False, num_workers=0)
    georgia_test = ecg_utils.load_test_data(constants.GEORGIA_TEST, args.domain_count, 1)
    georgia_test_dataloader = DataLoader(georgia_test, batch_size=dataloader_batchsize, shuffle=False, num_workers=0)
    ptb_test = ecg_utils.load_test_data(constants.PTB_TEST, args.domain_count, 2)
    ptb_test_dataloader = DataLoader(ptb_test, batch_size=dataloader_batchsize, shuffle=False, num_workers=0)
    print("Test CPSC data\n")
    test_main_and_domain_tasks(encoder, regressor, domain_predictor, cpsc_test_dataloader, constants.CPSC_TEST)
    print("\nTest Georgia data\n")
    test_main_and_domain_tasks(encoder, regressor, domain_predictor, georgia_test_dataloader, constants.GEORGIA_TEST)
    print("\nTest PTB data\n")
    test_main_and_domain_tasks(encoder, regressor, domain_predictor, ptb_test_dataloader, constants.PTB_TEST)
