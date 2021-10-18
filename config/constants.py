# paths for saving training output including trained models
LOSS_PATH = 'save_models/loss_pth'
PRE_TRAIN_ENCODER = 'save_models/pretrain_encoder'
PRE_TRAIN_REGRESSOR = 'save_models/pretrain_regressor'
PRE_TRAIN_DOMAIN = 'save_models/pretrain_domain'

PATH_ENCODER = 'save_models/encoder.pth'
PATH_REGRESSOR = 'save_models/regressor.pth'
PATH_DOMAIN = 'save_models/domain.pth'

CHK_PATH_ENCODER = 'save_models/encoder_chk.pth'
CHK_PATH_REGRESSOR = 'save_models/regressor_chk.pth'
CHK_PATH_DOMAIN = 'save_models/domain_chk.pth'

# all data file paths
CPSC_TRAIN = 'train_cpsc_xl'
CPSC_VAL = 'val_cpsc_xl'
CPSC_TEST = 'test_cpsc_xl'
GEORGIA_TRAIN = 'train_georgia'
GEORGIA_VAL = 'val_georgia'
GEORGIA_TEST = 'test_georgia'
PTB_TRAIN = 'train_ptb_xl'
PTB_VAL = 'val_ptb_xl'
PTB_TEST = 'test_ptb_xl'
CPSC_CLEAN_TRAIN = 'cpsc_clean_train'
CPSC_CLEAN_VAL = 'cpsc_clean_val'
CPSC_CLEAN_TEST = 'cpsc_clean_test'
CPSC_GAUS_TRAIN = 'cpsc_gaus_train'
CPSC_GAUS_VAL = 'cpsc_gaus_val'
CPSC_GAUS_TEST = 'cpsc_gaus_test'
CPSC_SIN_TRAIN = 'cpsc_sin_train'
CPSC_SIN_VAL = 'cpsc_sin_val'
CPSC_SIN_TEST = 'cpsc_sin_test'

data_split_template = 'data_splits/{}.csv'

# the paths to use for training the model
DOMAIN_A_TRAIN_PATH = data_split_template.format(CPSC_TRAIN)
DOMAIN_A_VAL_PATH = data_split_template.format(CPSC_VAL)
DOMAIN_B_TRAIN_PATH = data_split_template.format(GEORGIA_TRAIN)
DOMAIN_B_VAL_PATH = data_split_template.format(GEORGIA_VAL)
DOMAIN_C_TRAIN_PATH = data_split_template.format(PTB_TRAIN)
DOMAIN_C_VAL_PATH = data_split_template.format(PTB_VAL)

weights_file = 'config/weights.csv'
normal_class = '426783006'
label_file_dir = 'config/dx_mapping_scored.csv'
drop_classes = ['59118001', '63593006', '17338001']
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
results_filepath_template = 'results/{}_predictions.npz'
