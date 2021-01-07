import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from .resnet import resnet50


class Model(nn.Module):
    def __init__(
        self,
        last_conv_stride=1,
        last_conv_dilation=1,
        num_stripes=6,
        local_conv_out_channels=256,
        num_classes=0
    ):
        super(Model, self).__init__()

        self.base = resnet50(
            pretrained=False,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation,
            intermediate=True)
        self.num_stripes = num_stripes
        #print (self.base)

        self.local_conv_list = nn.ModuleList()
        for _ in range(num_stripes):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                #        nn.ReLU(inplace=False)
            ))

        # [BT,BT,C*2]->[BT,BT,D]
        self.relation_g = nn.Sequential(
            nn.Linear(local_conv_out_channels*num_stripes*2, 768),
            nn.ReLU(inplace=False),
            nn.Linear(768, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 512),
            nn.ReLU(inplace=False)
        )

        # [BT,D] -> [BT,1]
        self.relation_f = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 1),
            nn.ReLU(inplace=False)
        )

        # [B, CS, T] -> [B, 1, T]
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(local_conv_out_channels*num_stripes, 1, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=False),
        )

        if num_classes > 0:
            self.fc_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal(fc.weight, std=0.001)
                init.constant(fc.bias, 0)
                self.fc_list.append(fc)

    # feat: [B,T,C,H,W] -> local_feat: [B*T,c]

    def spatial_conv(self, feat):
        stripe_h = int(feat.size(-2) / self.num_stripes)
        local_feat_list = []
        feat = feat.view(-1, feat.size(-3), feat.size(-2),
                         feat.size(-1))  # [B*T, C, H, W]
        #print ('feat [BT,C,H,W]: '+str(feat.size()))
        for i in range(self.num_stripes):
            #[B*T, C, 1, 1]
            local_feat = F.avg_pool2d(
                feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                (stripe_h, feat.size(-1)))
            #print('feat [BT,C,1,1]: '+str(local_feat.size()))
            #[B*T, C, 1, 1]
            local_feat = self.local_conv_list[i](local_feat)
            #[B*T, C]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
        return local_feat_list

    # feat_map:[B,T,C,H,W]
    def get_temp_attention(self, feat_map, sample_mask):
        #print('Getting temp attention...')
        B, T = feat_map.size()[:2]
        # list of S features of shape [B*T, C]
        local_feat_list = self.spatial_conv(feat_map)
        # get temporal attention from feat_map
        stripe_cat = torch.cat(local_feat_list, dim=1)  # [B*T, C*S]
        #print('stripe concat ([B*T,C*S]): ' + str(stripe_cat.size()))
        stripe_cat = stripe_cat.view(B, T, -1)  # [B,T,C*S]
        #print('reshaped [B, T, C*S]: ' + str(stripe_cat.size()))
        stripe_cat = stripe_cat.permute(0, 2, 1)  # [B,C*S,T]
        #print('permutated [B, C*S, T]: ' + str(stripe_cat.size()))
        temp_attention = self.temporal_conv(stripe_cat)  # [B, 1, T]
        temp_attention = temp_attention.view(B, T)  # [B,T]
        # Normalize temp_attention
        temp_attention = F.sigmoid(temp_attention)
        temp_attention = temp_attention * sample_mask
        temp_attention = temp_attention / \
            (torch.sum(temp_attention, dim=1).view(B, 1))
        #print('temp_attention size after normalization' + str(temp_attention.size()))
        return temp_attention

    # feat_map:[B,T,C,H,W]
    def get_temp_attention_RN(self, feat_map, sample_mask):
        #print('Getting temp attention...')
        B, T = feat_map.size()[:2]
        # list of S features of shape [B*T, C]
        local_feat_list = self.spatial_conv(feat_map)
        # get temporal attention from feat_map
        stripe_cat = torch.cat(local_feat_list, dim=1)  # [B*T, C*S]
        stripe_cat = stripe_cat.view(B, T, -1)  # [B,T,C*S]
        #print('stripe concat ([B,T,C*S]): ' + str(stripe_cat.size()))
        feat_mat1 = stripe_cat[:, :, None, :]
        feat_mat1 = feat_mat1.repeat(
            1, 1, stripe_cat.size(1), 1)  # [B,T,T,C*S]
        #print('feat_mat1: \n'+str(feat_mat1.size()))
        feat_mat2 = stripe_cat[:, None, :, :]
        feat_mat2 = feat_mat2.repeat(
            1, stripe_cat.size(1), 1, 1)  # [B,T,T,C*S]
        #print('feat_mat2: \n'+str(feat_mat2.size()))
        feat_mat = torch.cat((feat_mat2, feat_mat1), dim=-1)  # [B,T,T,C*S*2]
        #print('feat matrix concatenated: \n'+str(feat_mat.size()))
        gout = self.relation_g(feat_mat)  # [B, T, T, D']
        #print('Gout: '+ str(gout.size()))
        gout = torch.sum(gout, dim=2)  # [B, T, D']
        #print('Gout sumed: '+ str(gout.size()))
        temp_attention = self.relation_f(gout)  # [B, T, 1]
        temp_attention = temp_attention.view(B, T)  # [B,T]
        # Normalize temp_attention
        temp_attention = F.sigmoid(temp_attention)
        temp_attention = temp_attention * sample_mask
        temp_attention = temp_attention / \
            (torch.sum(temp_attention, dim=1).view(B, 1))
        #print('temp_attention size after normalization' + str(temp_attention.size()))
        return temp_attention

    def forward(self, x, sample_mask=None):
        """
        Returns:
          local_feat_list: each member with shape [N, c]
          logits_list: each member with shape [N, num_classes]
        """
        # shape [N, C, H, W]
        if sample_mask is None:
            print('Sample_mask is None, exiting..')
            raise SystemExit
            _, _, _, feat = self.base(x)
        else:
            B = x.size(0)
            T = x.size(1)
            x_flat = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
            #print('sample_mask size: '+str(sample_mask.shape))

#      idx = torch.nonzero(sample_mask)
#      print sample_mask, idx
            output_layer1, output_layer2, output_layer3, output_layer4 = self.base(
                x_flat)
            #output_layer2: [B, T, C, H, W]
            output_layer2 = output_layer2.view(x.size(
                0), -1, output_layer2.size(-3), output_layer2.size(-2), output_layer2.size(-1))
            output_layer4 = output_layer4.view(x.size(
                0), -1, output_layer4.size(-3), output_layer4.size(-2), output_layer4.size(-1))
            # [B, T, C, H, W]
            output_layer2 = output_layer2 * sample_mask[:, :, None, None, None]
            output_layer4 = output_layer4 * sample_mask[:, :, None, None, None]
            #print('feature size before spatial conv: '+str(output_layer2.size()))
            # Get temp attention
            self_attention = self.get_temp_attention(
                output_layer4, sample_mask)
            coattention_RN = self.get_temp_attention_RN(
                output_layer4, sample_mask)
            temp_attention = (self_attention + coattention_RN)/2
            # resize in order to apply it
            temp_attention = temp_attention.view(B, T, 1, 1, 1)  # [B,T,1,1,1]
            # Apply temporal attention to feature from both layer2 and layer4
            output_layer2 = output_layer2.clone(
            ) * temp_attention  # [B,T,C,H,W]
            output_layer2 = torch.sum(output_layer2, dim=1).view(
                B, output_layer2.size(-3), output_layer2.size(-2), output_layer2.size(-1))  # [B,C,H,W]
            output_layer4 = output_layer4 * temp_attention
            output_layer4 = torch.sum(output_layer4, dim=1).view(
                B, output_layer4.size(-3), output_layer4.size(-2), output_layer4.size(-1))  # [B,C,H,W]
            # Get local_feat_list from both layer2 and layer4
            output_layer2 = self.base(output_layer2, from_layer='layer2')
            local_feat_l2 = self.spatial_conv(output_layer2)
            local_feat_l4 = self.spatial_conv(output_layer4)
            # Get a weighted average of local_feat from l2 and l4
            local_feat_list = []
            for i in range(self.num_stripes):
                local_feat_list.append(
                    local_feat_l2[i]*0.5+local_feat_l4[i]*0.5)
            #print('Final local_feat size: '+str(local_feat_list[0].size()))

        logits_list = []
        if hasattr(self, 'fc_list'):
            for i in range(self.num_stripes):
                logits_list.append(self.fc_list[i](local_feat_list[i]))
            return local_feat_list, logits_list
        return local_feat_list
