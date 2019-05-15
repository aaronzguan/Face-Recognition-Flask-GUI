import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from common.util import L2Norm

import sys


class softmax(nn.Module):
    def __init__(self, class_num, args, bias=False):
        super(softmax, self).__init__()
        self.in_features = args.feature_dim
        self.out_features = class_num
        self.w_norm = args.use_w_norm
        self.f_norm = args.use_f_norm
        self.s = args.s
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input, target):
        if self.w_norm:
            weight = L2Norm()(self.weight)
        else:
            weight = self.weight
        if self.f_norm:
            x = L2Norm()(input)
        else:
            x = input
        scores = F.linear(x, weight, self.bias)  # x @ weight.t() + bias(if any)
        if self.w_norm and self.f_norm:
            assert self.s > 1.0, 'scaling factor s should > 1.0'
            scores_new = self.s * scores
        else:
            scores_new = scores
        return scores, self.CELoss(scores_new, target.view(-1))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class focalloss(nn.Module):
    def __init__(self, class_num, args, gamma=5., bias=False):
        super(focalloss, self).__init__()
        self.in_features = args.feature_dim
        self.out_features = class_num
        self.w_norm = args.use_w_norm
        self.f_norm = args.use_f_norm
        self.s = args.s
        self.gamma = gamma
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input, target):
        if self.w_norm:
            weight = L2Norm()(self.weight)
        else:
            weight = self.weight
        if self.f_norm:
            x = L2Norm()(input)
        else:
            x = input
        scores = F.linear(x, weight, self.bias)  # x @ weight.t() + bias(if any)
        if self.w_norm and self.f_norm:
            assert self.s > 1.0, 'scaling factor s should > 1.0'
            scores_new = self.s * scores
        else:
            scores_new = scores
        softmax = F.softmax(scores_new, dim=1)
        target_mask = torch.zeros_like(scores_new).scatter_(1, target, 1)
        loss = -1 * target_mask * torch.log(softmax) # cross entropy
        loss = loss * (1 - softmax) ** self.gamma # focal loss
        return scores, loss.sum(dim=1).mean()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class asoftmax(nn.Module):
    def __init__(self, class_num, args):
        super(asoftmax, self).__init__()
        self.in_features = args.feature_dim
        self.out_features = class_num
        self.w_norm = args.use_w_norm
        self.f_norm = args.use_f_norm
        self.m = args.m_1
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.Lambda = 1500.0
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.register_parameter('bias', None)
        self.reset_parameters() #weight initialization
        assert (self.w_norm == True and self.f_norm == False), 'Wrong implementation of A-Softmax loss.'
        assert self.m >= 1., 'margin m of asoftmax should >= 1.0'


    def forward(self, input, target):
        if self.w_norm:
            weight = L2Norm()(self.weight)
        else:
            weight = self.weight
        scores = F.linear(input, weight, self.bias)  # x @ weight.t() + bias(if any)
        index = torch.zeros_like(scores).scatter_(1, target, 1)

        x_len = input.norm(dim=1)
        cos_theta = scores / (x_len.view(-1, 1).clamp(min=1e-12))
        cos_theta = cos_theta.clamp(-1, 1)
        m_theta = self.m * torch.acos(cos_theta)
        k = (m_theta / 3.141592653589793).floor().detach()
        cos_m_theta = torch.cos(m_theta)
        psi_theta = ((-1)**k) * cos_m_theta - 2*k
        psi_theta = psi_theta * x_len.view(-1, 1)

        self.Lambda = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        self.it += 1
        scores_new = scores - scores*index/(1+self.Lambda) + psi_theta*index/(1+self.Lambda)
        return scores, self.CELoss(scores_new, target.view(-1))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class amsoftmax(nn.Module):
    def __init__(self, class_num, args):
        super(amsoftmax, self).__init__()
        self.in_features = args.feature_dim
        self.out_features = class_num
        self.w_norm = args.use_w_norm
        self.f_norm = args.use_f_norm
        self.s = args.s
        self.m = args.m_3
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.register_parameter('bias', None)
        self.reset_parameters() #weight initialization
        assert (self.w_norm and self.f_norm), 'Wrong implementation of AMSoftmax loss.'
        assert self.s > 1.0, 'scaling factor s should > 1.0'
        assert self.m > 0., 'scaling factor s should > 1.0'


    def forward(self, input, target):
        if self.w_norm:
            weight = L2Norm()(self.weight)
        else:
            weight = self.weight
        if self.f_norm:
            x = L2Norm()(input)
        else:
            x = input
        scores = F.linear(x, weight, self.bias)  # x @ weight.t() + bias(if any)
        index = torch.zeros_like(scores).scatter_(1, target, 1)
        scores_new = self.s*(scores - scores*index + (scores - self.m)*index)
        # scores_new = input_norm*(scores - scores*index + (scores - self.m)*index)
        return scores, self.CELoss(scores_new, target.view(-1))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


class centerloss(nn.Module):
    def __init__(self, class_num, args, bias=False):
        super(centerloss, self).__init__()
        self.device = args.device
        self.lamb = args.lamb  # weight of center loss
        self.alpha = 0.5  # weight of updating centers
        self.in_features = args.feature_dim
        self.class_num = class_num
        self.f_norm = args.use_f_norm
        self.centers = torch.nn.Parameter(torch.Tensor(self.class_num, self.in_features))
        self.centers.requires_grad = False
        self.delta_centers = torch.zeros_like(self.centers)
        self.softmaxloss = softmax(class_num, args)
        self.reset_parameters()

    def forward(self, input, target):
        scores, loss = self.softmaxloss(input, target)  # Softmax loss
        '''
            Center loss: follow the paper's implementation.
            Inspired by https://github.com/louis-she/center-loss.pytorch/blob/5be899d1f622d24d7de0039dc50b54ce5a6b1151/loss.py
        '''
        ## Center loss
        if self.f_norm:
            x = L2Norm()(input)
        else:
            x = input
        self.update_center(x, target)

        target_centers = self.centers[target].squeeze()
        center_loss = ((x - target_centers) ** 2).sum(dim=1).mean()
        return scores, loss + self.lamb * 0.5 * center_loss

    def update_center(self, features, targets):
        # implementation equation (4) in the center-loss paper
        targets, indices = torch.sort(targets.view(-1))
        target_centers = self.centers[targets]
        features = features.detach()[indices]
        delta_centers = target_centers - features
        uni_targets, indices = torch.unique(targets.cpu(), sorted=True, return_inverse=True)
        uni_targets = uni_targets.to(self.device)
        indices = indices.to(self.device)
        delta_centers = torch.zeros(uni_targets.size(0), delta_centers.size(1)).to(self.device).index_add_(0, indices, delta_centers)
        targets_repeat_num = uni_targets.size()[0]
        uni_targets_repeat_num = targets.size()[0]
        targets_repeat = targets.repeat(targets_repeat_num).view(targets_repeat_num, -1)
        uni_targets_repeat = uni_targets.unsqueeze(1).repeat(1, uni_targets_repeat_num)
        same_class_feature_count = torch.sum(targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)
        delta_centers = delta_centers / (same_class_feature_count + 1.0) * self.alpha
        result = torch.zeros_like(self.centers)
        result[uni_targets, :] = delta_centers
        self.centers -= result

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))



class TriSoftmaxLoss(nn.Module):
    def __init__(self, feature_dim, class_num, w_norm, f_norm, s=30, m=0.4):
        super(TriSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.margin = 0.5
        self.w_norm = w_norm
        self.f_norm = f_norm
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.Tensor(feature_dim, class_num))
        self.weight.data.uniform_(-1, 1)
        self.step = 0
        self.step_max = 5000

    def forward(self, input, target, is_feature=False):
        if self.w_norm:
            weight = L2Norm()(self.weight, dim=0)
        else:
            weight = self.weight
        if self.f_norm:
            x = L2Norm()(input)
        else:
            x = input
        scores = x @ weight  #B*class_num
        index = torch.zeros_like(scores).scatter_(1, target, 1)
        scores_new = self.s*(scores - scores*index + (scores - self.m)*index)

        # triplet loss
        scores_without_pos = scores - scores*index - 1.0*index
        _, min_neg  = scores_without_pos.topk(k=1)
        index_neg = torch.zeros_like(scores).scatter_(1, min_neg, 1)
        d_a_p = 1 - (scores*index).sum(dim=1)
        d_a_n = 1 - (scores*index_neg).sum(dim=1)
        loss = torch.clamp(self.margin + d_a_p - d_a_n, min=0.0)
        self.step += 1

        if is_feature:
            return scores, scores_new
        else:
            return scores, self.CELoss(scores_new, target.view(-1)) + loss.mean()


class AngularLoss(nn.Module):
    def __init__(self, class_num, w_norm, f_norm):
        super(AngularLoss, self).__init__()
        self.w_norm = w_norm
        self.f_norm = f_norm
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.inner_product = nn.Linear(512, class_num, bias=False)

    def forward(self, input, targets):
        if self.w_norm:
            self.inner_product.weight.data = L2Norm()(self.inner_product.weight.data)
        if self.f_norm:
            input = L2Norm()(input)
        # normalize the score to [0, 1]
        scores = (self.inner_product(input) + 1.0) * 0.5 * 50
        log_scores = torch.log(scores)
        log_scores.clamp(min=1e-12)
        return scores, self.CELoss(log_scores, targets.view(-1))


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.alpha = 1

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F, Classnum) F=in_features Classnum=out_features
        xlen = x.norm(dim=1)
        wlen = w.norm(dim=0)
        w_norm = w / (wlen.view(1, -1) + 1e-8)
        dot_product = x @ w_norm  # size=(B,Classnum)
        cos_theta = dot_product / (xlen.view(-1, 1) + 1e-8)
        cos_theta = cos_theta.clamp(-1,1)
        m_theta = self.m * torch.acos(cos_theta)
        k = (m_theta / 3.141592653589793).floor().detach()
        cos_m_theta = torch.cos(m_theta)
        psi_theta = (-1)**k*cos_m_theta - 2*k
        return psi_theta*xlen.view(-1, 1), dot_product # size=(B,Classnum,2)


class ASoftmaxLoss(nn.Module):
    def __init__(self):
        super(ASoftmaxLoss, self).__init__()
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.Lambda = 1500.0
        self.CELoss = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        self.Lambda = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        # input: (dot_m_prodct, dot_product)
        modified_dot_product, dot_product = input
        index = torch.zeros_like(dot_product).scatter_(1, target, 1)
        dot_product = dot_product  + ((modified_dot_product - dot_product) * index) / (1 + self.Lambda)
        #self.Lambda = max(self.LambdaMin, self.Lambda - 1)
        self.it += 1
        return self.CELoss(dot_product, target.view(-1))