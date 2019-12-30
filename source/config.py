from easydict import EasyDict as edict

config = edict()
config.SV_TRAIN = edict()
config.USV_TRAIN = edict()
config.IMG = edict()
config.SV_TEST = edict()
config.USV_TEST = edict()

# CIFAR10 Image Parameters
config.IMG.height = 32
config.IMG.width = 32
config.IMG.num_classes = 10

# Hyper Parameters
# Supervised Training
config.SV_TRAIN.num_epoches = 100
config.SV_TRAIN.batch_size = 100
config.SV_TRAIN.dropout = 0.5
config.SV_TRAIN.learning_rate_init = 1e-3
config.SV_TRAIN.lr_start_decay = 79
config.SV_TRAIN.lr_decay = 0.5
config.SV_TRAIN.lr_decay_period = 10
config.SV_TRAIN.logs_train = '..\\logs\\supervised\\logs_sv.txt'
config.SV_TRAIN.model_dir = '..\\model\\supervised\\sv_model.ckpt'
config.SV_TRAIN.logs_dir = '..\\logs\\supervised'

# Unsupervised Training
config.USV_TRAIN.labeled_num_epoches = 120
config.USV_TRAIN.unlabeled_num_epoches = 60
config.USV_TRAIN.batch_size = 100
config.USV_TRAIN.dropout = 0.5
config.USV_TRAIN.learning_rate_init = 1e-3
config.USV_TRAIN.lr_start_decay = 120+29
config.USV_TRAIN.lr_decay = 0.5
config.USV_TRAIN.lr_decay_period = 10
config.USV_TRAIN.logs_train = '..\\logs\\mean_teacher\\logs_usv.txt'
config.USV_TRAIN.model_dir = '..\\model\\mean_teacher\\usv_model.ckpt'
config.USV_TRAIN.logs_dir = '..\\logs'

# Test
config.SV_TEST.logs_test = '..\\logs\\supervised\\logs_svtest.txt'
config.USV_TEST.logs_test = '..\\logs\\mean_teacher\\logs_usvtest.txt'

