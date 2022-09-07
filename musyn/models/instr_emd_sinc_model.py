import os
from collections import OrderedDict

import torch
import torch.nn as nn

from musyn.models.backbones import resnet34, se_resnet34
from musyn.models.base_model import BaseModel
from musyn.models.heads import LinearClsHead, AngularClsHead
from musyn.models.necks import LDE
from musyn.models.transforms import SincConv


class InstrEmdSincModel(BaseModel):
    """Embedding Net (Instrument Embedding SincConv Model) 
    """
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        # tranfer raw audio to sinc feature
        kernel_size = int(opt.transform.kernel_size / 1000 * opt.transform.sr[0])
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1 
        stride = int(opt.transform.stride / 1000 * opt.transform.sr[0])
        self.trans = SincConv(
            out_channels=opt.transform.out_channels,
            kernel_size=kernel_size,
            in_channels=opt.transform.in_channels,
            padding=opt.transform.padding,
            stride=stride,
            init_type=opt.transform.init_type,
            min_low_hz=opt.transform.min_low_hz,
            min_band_hz=opt.transform.min_band_hz,
            requires_grad=opt.transform.requires_grad,
        )

        # encode sinc feature to latent space
        if opt.backbone.type == 'se_resnet34':
            self.encoder = se_resnet34()
            _feature_dim = 128
        elif opt.backbone.type == 'resnet34':
            self.encoder = resnet34()
            _feature_dim = 128
        else:
            raise NotImplementedError

        if os.path.isfile(opt.backbone.pretrained):
            checkpoint = torch.load(opt.backbone.pretrained, 
                map_location=lambda storage, loc: storage)
            new_ckpt = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'encoder.' in k:
                    new_k = k.replace('encoder.', '')
                    new_ckpt[new_k] = v
            self.encoder.load_state_dict(new_ckpt, strict=False)

        self.pool = LDE(
            D=opt.neck.D,
            input_dim=_feature_dim,
            pooling=opt.neck.pooling,
            network_type=opt.neck.network_type,
            distance_type=opt.neck.distance_type
        )
        
        if opt.neck.pooling=='mean':
            in_channels = _feature_dim*opt.neck.D
        if opt.neck.pooling=='mean+std':
            in_channels = _feature_dim*2*opt.neck.D

        self.fc0 = nn.Linear(in_channels, opt.head1.hidden_dim)
        self.bn0  = nn.BatchNorm1d(opt.head1.hidden_dim)

        if opt.head1.type == 'AngularClsHead':
            self.fc11 = AngularClsHead(
                num_classes=opt.head1.num_classes,
                in_channels=opt.head1.hidden_dim,
                m=opt.head1.m,
            )
        elif opt.head1.type == 'LinearClsHead':
            self.fc11 = LinearClsHead(
                num_classes=opt.head1.num_classes,
                in_channels=opt.head1.hidden_dim,
            )
        else:
            raise NotImplementedError
        
        if opt.head2.type == 'AngularClsHead':
            self.fc12 = AngularClsHead(
                num_classes=opt.head2.num_classes,
                in_channels=opt.head2.hidden_dim,
                m=opt.head2.m,
            )
        elif opt.head2.type == 'LinearClsHead':
            self.fc12 = LinearClsHead(
                num_classes=opt.head2.num_classes,
                in_channels=opt.head2.hidden_dim,
            )
        else:
            raise NotImplementedError


    def forward(self, x):
        x = x.transpose(1, -1) # batch * time * channel
        x = self.trans(x)
        x = self.encoder(x)
        x = self.pool(x)

        feat = self.fc0(x)
        feat_bn = self.bn0(feat)

        out1 = self.fc11(feat_bn)
        out2 = self.fc12(feat_bn)
        return feat, out1, out2


    def predict(self, x):
        x = x.transpose(1, -1) # batch * time * channel
        x = self.trans(x)
        x = self.encoder(x)
        x = self.pool(x)
        if type(x) is tuple:
            x = x[0] 
        feat = self.fc0(x)
        return feat



