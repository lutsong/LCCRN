import torch
import sys
sys.path.append('../../../../')
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import Conv_4, ResNet
from numpy import random
import scipy.misc
from PIL import Image
from models.LCEM import LCEM, SelfCorrelationComputation

# from  vs import  tensor2im, merge ,save_img

class LCCRN(nn.Module):

    def __init__(self, way=None, shots=None, resnet=False, is_pretraining=False, num_cat=None):

        super().__init__()

        if resnet:
            num_channel = 640

            self.feature_extractor_1 = ResNet.resnet12()
            self.feature_extractor_2 = ResNet.resnet12()
            self.Woodbury = False
        else:
            num_channel = 64

            self.feature_extractor_1 = Conv_4.BackBone(num_channel)
            self.feature_extractor_2 = Conv_4.BackBone(num_channel)
            self.Woodbury = True
        self.lcem_module = self._make_lcem_layer(planes=[640, 64, 64, 64, 640])
        self.shots = shots
        self.way = way
        self.resnet = resnet

        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel

        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.W = nn.Parameter(torch.full((4, 1), 1. / 4), requires_grad=True)
        self.amplitude = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)

        # H*W=5*5=25, resolution of feature map, correspond to r in the paper
        self.resolution = 25
        # correpond to [alpha, beta] in the paper
        # if is during pre-training, we fix them to 0
        self.r = nn.Parameter(torch.zeros(8), requires_grad=not is_pretraining)
        #self.k = nn.Parameter(torch.FloatTensor([0.01]), requires_grad=True)
        if is_pretraining:
            # number of categories during pre-training
            self.num_cat = num_cat
            # category matrix, correspond to matrix M of section 3.6 in the paper
            self.cat_mat_1 = nn.Parameter(torch.randn(self.num_cat, self.resolution, self.d), requires_grad=True)
            self.cat_mat_2 = nn.Parameter(torch.randn(self.num_cat, self.resolution, self.d), requires_grad=True)
    def _make_lcem_layer(self, planes):
        # planes=[640, 64, 64, 64, 640]
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()
        corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
        self_block = LCEM(planes=planes, stride=stride)
        layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)
    def get_feature_map(self, inp,pre_trian):
        if pre_trian:
            feature_map_1 = self.feature_extractor_1(inp)
            feature_map_2 = self.feature_extractor_2(inp)
        else:
            feature_map_1 = self.feature_extractor_1(inp)
            feature_map_1=F.normalize(feature_map_1, dim=1, p=2)
            feature_map_2 = self.feature_extractor_2(inp)
            feature_map_2=F.normalize(feature_map_2, dim=1, p=2)
        return feature_map_1, feature_map_2  # N,HW,C

    def get_recon_dist(self, query, support, alpha, beta, Woodbury):
        # query: way*query_shot*resolution, d
        # support: way, shot*resolution , d
        # Woodbury: whether to use the Woodbury Identity as the implementation or not

        # correspond to kr/d in the paper
        reg = support.size(1) / support.size(2)

        # correspond to lambda in the paper
        lam = reg * alpha.exp() + 1e-6

        # correspond to gamma in the paper
        rho = beta.exp()

        st = support.permute(0, 2, 1)  # way, d, shot*resolution

        if Woodbury:
            # correspond to Equation 10 in the paper

            sts = st.matmul(support)  # way, d, d
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()  # way, d, d
            hat = m_inv.matmul(sts)  # way, d, d

        else:
            # correspond to Equation 8 in the paper

            sst = support.matmul(st)  # way, shot*resolution, shot*resolution
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(
                lam)).inverse()  # way, shot*resolution, shot*resolutionsf
            hat = st.matmul(m_inv).matmul(support)  # way, d, d

        Q_bar = query.matmul(hat).mul(rho)  # way, way*query_shot*resolution, d
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)  # way*query_shot*resolution, way
        return dist

    def get_neg_l2_dist(self, inp, way, shot, query_shot, return_support=False, Woodbury=True):
        batch_size = inp.size(0)
        resolution = self.resolution
        d = self.d
        alpha = self.r[:4]
        beta = self.r[4:]
        pre_train=False
        feature_map_1, feature_map_2 = self.get_feature_map(inp,pre_train)
        feature_map_1 = feature_map_1.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()
        self_map = self.lcem_module(feature_map_2)
        #self_map = F.relu(self_map)
        self_map = F.normalize(self_map, dim=1, p=2)
        feature_map_2 = self_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()
        support_1 = feature_map_1[:way * shot].view(way, shot * resolution, d)
        query_1 = feature_map_1[way * shot:].view(way * query_shot * resolution, d)
        support_2 = feature_map_2[:way * shot].view(way, shot * resolution, d)
        query_2 = feature_map_2[way * shot:].view(way * query_shot * resolution, d)
        recon_dist_1 = self.get_recon_dist(query=query_1, support=support_1, alpha=alpha[0],
                                           beta=beta[0], Woodbury=Woodbury)  # way*query_shot*resolution, way
        recon_dist_2 = self.get_recon_dist(query=query_2, support=support_2, alpha=alpha[1],
                                           beta=beta[1], Woodbury=Woodbury)
        recon_dist_3 = self.get_recon_dist(query=query_1, support=support_2, alpha=alpha[2],
                                           beta=beta[2], Woodbury=Woodbury)
        recon_dist_4 = self.get_recon_dist(query=query_2, support=support_1, alpha=alpha[3],
                                           beta=beta[3], Woodbury=Woodbury)
        # way*query_shot*resolution, way
        neg_l2_dist_1 = recon_dist_1.neg().view(way * query_shot, resolution, way).mean(1)  # way*query_shot, way
        neg_l2_dist_2 = recon_dist_2.neg().view(way * query_shot, resolution, way).mean(1)
        neg_l2_dist_3 = recon_dist_3.neg().view(way * query_shot, resolution, way).mean(1)
        neg_l2_dist_4 = recon_dist_4.neg().view(way * query_shot, resolution, way).mean(1)
        neg_l2_dist = self.W[0] * neg_l2_dist_1 + self.W[1] * neg_l2_dist_2 + self.W[2] * neg_l2_dist_3 + self.W[
            3] * neg_l2_dist_4

        if return_support:
            return neg_l2_dist, support_1
        else:
            return neg_l2_dist

    def meta_test(self, inp, way, shot, query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot
                                           )

        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index

    """
    def forward_pretrain(self, inp):

        alpha = self.r[:4]
        beta = self.r[4:]
        pre_train=True
        feature_map_1, feature_map_2 = self.get_feature_map(inp,pre_train)
        # N = self.amplitude * torch.randn(feature_map_2.size()[-2:]).cuda()
        # N_rule=F.relu(N)
        # Nl =  N_rule.unsqueeze(0).repeat( feature_map_2[:way * shot].size(0), 1, 1)
        batch_size = feature_map_1.size(0)
        feature_map_1 = feature_map_1.view(batch_size * self.resolution, self.d)
        feature_map_2 = feature_map_2.view(batch_size * self.resolution, self.d)
        recon_dist_1 = self.get_recon_dist(query=feature_map_1, support=self.cat_mat_1, alpha=alpha[0],
                                           beta=beta[0], Woodbury=self.Woodbury)  # way*query_shot*resolution, way
        recon_dist_2 = self.get_recon_dist(query=feature_map_2, support=self.cat_mat_2, alpha=alpha[1],
                                           beta=beta[1], Woodbury=self.Woodbury)
        recon_dist_3 = self.get_recon_dist(query=feature_map_1, support=self.cat_mat_2, alpha=alpha[2],
                                           beta=beta[2], Woodbury=self.Woodbury)
        recon_dist_4 = self.get_recon_dist(query=feature_map_2, support=self.cat_mat_1, alpha=alpha[3],
                                           beta=beta[3], Woodbury=self.Woodbury)

        neg_l2_dist_1 = recon_dist_1.neg().view(batch_size, self.resolution, self.num_cat).mean(1)  # way*query_shot, way
        neg_l2_dist_2 = recon_dist_2.neg().view(batch_size, self.resolution, self.num_cat).mean(1)
        neg_l2_dist_3 = recon_dist_3.neg().view(batch_size, self.resolution, self.num_cat).mean(1)
        neg_l2_dist_4 = recon_dist_4.neg().view(batch_size, self.resolution, self.num_cat).mean(1)
        neg_l2_dist = self.W[0] * neg_l2_dist_1 + self.W[1] * neg_l2_dist_2 + self.W[2] * neg_l2_dist_3 + self.W[
            3] * neg_l2_dist_4


        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction
"""
    def forward(self, inp):

        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True,
                                                    Woodbury=self.Woodbury,
                                                    )

        logits = neg_l2_dist * self.scale

        log_prediction = F.log_softmax(logits, dim=1)
        return log_prediction, support
