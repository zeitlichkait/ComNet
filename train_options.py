"""Training options."""

import argparse

parser = argparse.ArgumentParser(description="ComN")

# Data.
parser.add_argument('--data-name',
                    type=str,
                    choices=['ucf101', 'hmdb51'],
                    help='dataset name.')
parser.add_argument('--data-root',
                    type=str,
                    default='/mnt/data/ar-datasets/ucf101_hevc_features/',
                    help='root of data directory.')
parser.add_argument('--train-list',
                    type=str,
                    default='ucf101_train_list1.txt',
                    help='training example list.')
parser.add_argument('--test-list',
                    type=str,
                    default='ucf101_test_list1.txt',
                    help='testing example list.')

# Model.
parser.add_argument('--representation',
                    type=str,
                    choices=['iframe', 'mv', 'residual'],
                    help='data representation.')
parser.add_argument('--arch',
                    type=str,
                    default="resnet18",
                    help='base architecture.')
parser.add_argument('--num_segments',
                    type=int,
                    default=3,
                    help='number of TSN segments.')
parser.add_argument('--no-pretrained',
                    action='store_true',
                    help='disable pretrained.')
parser.add_argument('--no-prepossesing',
                    action='store_true',
                    help='disable prepossesing.')

# Training.
parser.add_argument('--epochs',
                    default=500,
                    type=int,
                    help='number of training epochs.')
parser.add_argument('--batch-size', default=40, type=int, help='batch size.')
parser.add_argument('--batch-size-val',
                    default=32,
                    type=int,
                    help='batch size.')
parser.add_argument('--lr',
                    default=0.001,
                    type=float,
                    help='base learning rate.')
parser.add_argument('--lr-steps',
                    default=[200, 300, 400],
                    type=float,
                    nargs="+",
                    help='epochs to decay learning rate.')
parser.add_argument('--lr-decay',
                    default=0.1,
                    type=float,
                    help='lr decay factor.')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    help='weight decay.')

# Log.
parser.add_argument('--eval-freq',
                    default=5,
                    type=int,
                    help='evaluation frequency (epochs).')
parser.add_argument('--workers',
                    default=8,
                    type=int,
                    help='number of data loader workers.')
parser.add_argument('--model-prefix',
                    type=str,
                    required=True,
                    help="prefix of model name.")
parser.add_argument('--gpus', type=str, required=True, help='gpu ids.')
