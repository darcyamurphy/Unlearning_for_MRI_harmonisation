# paths for saving training output including trained models
LOSS_PATH = 'loss_pth'
PRE_TRAIN_ENCODER = 'save_models/pretrain_encoder'
PRE_TRAIN_REGRESSOR = 'save_models/pretrain_regressor'
PRE_TRAIN_DOMAIN = 'save_models/pretrain_domain'

PATH_ENCODER = 'save_models/encoder_pth'
PATH_REGRESSOR = 'save_models/regressor_pth'
PATH_DOMAIN = 'save_models/domain_pth'

CHK_PATH_ENCODER = 'save_models/encoder_chk_pth'
CHK_PATH_REGRESSOR = 'save_models/regressor_chk_pth'
CHK_PATH_DOMAIN = 'save_models/domain_chk_pth'

# all data file paths
CPSC_TRAIN = 'data_splits/train_cpsc_xl.csv'
CPSC_VAL = 'data_splits/val_cpsc_xl.csv'
CPSC_TEST = 'data_splits/test_cpsc_xl.csv'
GEORGIA_TRAIN = 'data_splits/train_georgia.csv'
GEORGIA_VAL = 'data_splits/val_georgia.csv'
GEORGIA_TEST = 'data_splits/test_georgia.csv'
PTB_TRAIN = 'data_splits/train_ptb_xl.csv'
PTB_VAL = 'data_splits/val_ptb_xl.csv'
PTB_TEST = 'data_splits/test_ptb_xl.csv'
CPSC_CLEAN_TRAIN = 'data_splits/cpsc_clean_train.csv'
CPSC_CLEAN_VAL = 'data_splits/cpsc_clean_val.csv'
CPSC_CLEAN_TEST = 'data_splits/cpsc_clean_test.csv'
CPSC_GAUS_TRAIN = 'data_splits/cpsc_gaus_train.csv'
CPSC_GAUS_VAL = 'data_splits/cpsc_gaus_val.csv'
CPSC_GAUS_TEST = 'data_splits/cpsc_gaus_test.csv'
CPSC_SIN_TRAIN = 'data_splits/cpsc_sin_train.csv'
CPSC_SIN_VAL = 'data_splits/cpsc_sin_val.csv'
CPSC_SIN_TEST = 'data_splits/cpsc_sin_test.csv'

# the paths to use for training the model
DOMAIN_A_TRAIN_PATH = CPSC_TRAIN
DOMAIN_A_VAL_PATH = CPSC_VAL
DOMAIN_B_TRAIN_PATH = GEORGIA_TRAIN
DOMAIN_B_VAL_PATH = GEORGIA_VAL
DOMAIN_C_TRAIN_PATH = PTB_TRAIN
DOMAIN_C_VAL_PATH = PTB_VAL

weights_file = 'config/weights.csv'
normal_class = '426783006'
label_file_dir = 'config/dx_mapping_scored.csv'
drop_classes = ['59118001', '63593006', '17338001']
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
