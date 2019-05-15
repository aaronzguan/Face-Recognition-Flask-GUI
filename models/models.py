import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
torch.backends.cudnn.benchmark = True

import os
from collections import OrderedDict
from . import networks, losses

def weights_init(m, type='xavier'):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        if type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif type == 'orthogonal':
            nn.init.orthogonal_(m.weight)
        elif type == 'gaussian':
            m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CreateModel(nn.Module):
    def __init__(self, args):
        super(CreateModel, self).__init__()
        self.args = args
        self.feature_dim = args.feature_dim
        self.device = args.device
        self.gpu_ids = args.gpu_ids
        if 'spherenet' in args.backbone:
            num_layers = int(args.backbone.split('spherenet')[-1])
            self.backbone = getattr(networks, 'spherenet')(num_layers, args.feature_dim, args.use_pool, args.use_dropout, args.use_lbp)
        else:
            self.backbone = getattr(networks, args.backbone)(args.feature_dim, args.use_pool, args.use_dropout)
        self.backbone.to(self.device)
        # from torchsummary import summary
        # print(summary(self.backbone, (3, 112, 96)))

    def train_setup(self, class_num):
        self.model_names = ['backbone', 'criterion']
        self.loss_names = ['loss_ce', 'lr']

        ## Setup nn.DataParallel if necessary
        if len(self.gpu_ids) > 1:
            self.backbone = nn.DataParallel(self.backbone)

        ## Objective function
        self.criterion = getattr(losses, self.args.loss_type)
        self.criterion = self.criterion(class_num, self.args)
        self.criterion.to(self.device)

        ## Setup optimizer
        self.lr = self.args.lr
        self.save_dir = os.path.join(self.args.checkpoints_dir, self.args.name)
        params = list(self.backbone.parameters()) + list(self.criterion.parameters())
        self.optimizer = optim.SGD(params, lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.decay_steps, gamma=0.1)

        ## Weight initialization
        # self.backbone.apply(weights_init)
        # self.criterion.apply(weights_init)

        ## Switch to training mode
        self.train()

    def update_learning_rate(self):
        self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

    def optimize_parameters(self, data):
        input, target = data
        self.score, self.loss_ce = self.forward(input.to(self.device), target.to(self.device))
        self.optimizer.zero_grad()
        self.loss_ce.backward()
        self.optimizer.step()

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # float(...) works for both scalar tensor and float number
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)
                if len(self.gpu_ids) > 1 and torch.cuda.is_available():
                    try:
                        torch.save(net.module.cpu().state_dict(), save_path)
                    except:
                        torch.save(net.cpu().state_dict(), save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def forward(self, input, target=None, is_feature=False):
        features = self.backbone(input)
        if is_feature:
            return features
        else:
            return self.criterion(features, target)

    def test(self):
        for name in self.model_names:
            try:
                if isinstance(name, str):
                    getattr(self, name).eval()
            except:
                print('{}.eval() cannot be implemented as {} does not exist.'.format(name, name))

    def train(self):
        for name in self.model_names:
            try:
                if isinstance(name, str):
                    getattr(self, name).train()
            except:
                print('{}.train() cannot be implemented as {} does not exist.'.format(name, name))