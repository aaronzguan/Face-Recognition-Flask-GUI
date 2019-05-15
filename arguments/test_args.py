from . import common_args, modify_args
from common.util import str2bool

def get_args():
    parser = common_args.get_args()
    parser.add_argument('--isTrain', default=False, type=str2bool, help='is train?')
    args = modify_args.run(parser, dev=True)
    return args



