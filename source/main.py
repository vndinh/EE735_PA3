from sv_train import supervised_training
from usv_train import unsupervised_training
from sv_test import supervised_testing
from usv_test import unsupervised_testing

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='svtrain', help='Running process')
  args = parser.parse_args()

  if args.mode == 'svtrain':
    supervised_training()
  elif args.mode == 'usvtrain':
    unsupervised_training()
  elif args.mode == 'svtest':
    supervised_testing()
   elif args.mode == 'usvtest':
   	unsupervised_testing()
  else:
    raise Exception("Unknown mode")