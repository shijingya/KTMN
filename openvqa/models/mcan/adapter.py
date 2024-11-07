# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask
from openvqa.models.mcan.geometric import absolutePostion, PositionEmbeddingLearned, PositionEmbeddingSine

class Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Adapter, self).__init__(__C)
        self.__C = __C

    def bbox_proc(self, bbox):
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])
        return torch.cat((bbox, area.unsqueeze(2)), -1)

    def vqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]   # 2048
        '''
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
         '''
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)   # 图像特征[100,2048]--->[100,512]
        self.linear_abs = nn.Linear(5, __C.HIDDEN_SIZE)    # 用于映射边界框5维特征
        self.relu_1 = nn.ReLU(inplace=True)


    def gqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)

        if __C.USE_AUX_FEAT:
            self.grid_linear = nn.Linear(__C.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)

    def clevr_init(self, __C):
        self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)  # (1024,512)
        self.abs_position_embedding = PositionEmbeddingLearned(256)
        self.grid_abs = PositionEmbeddingSine(256, normalize=True)

    def vqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        w_feat = feat_dict['W_FEAT']
        h_feat = feat_dict['H_FEAT']
        w_feat = w_feat.unsqueeze(1)
        h_feat = h_feat.unsqueeze(1)

        wh_feat = torch.cat((w_feat, h_feat), dim=-1)

        img_feat_mask = make_mask(frcn_feat)  # 64,1,1,100
        '''
        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        '''
        img_feat = self.frcn_linear(frcn_feat)  # 64,100,512
        region_try = absolutePostion(bbox_feat, wh_feat)  # (bs, 100, 5)
        region_try = self.linear_abs(region_try)  # 64,100,512
        region_try = self.relu_1(region_try)
        # region_try = None
        return img_feat, img_feat_mask, region_try


    def gqa_forward(self, feat_dict):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(frcn_feat)

        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)

        if self.__C.USE_AUX_FEAT:
            grid_feat_mask = make_mask(grid_feat)
            img_feat_mask = torch.cat((img_feat_mask, grid_feat_mask), dim=-1)
            grid_feat = self.grid_linear(grid_feat)
            img_feat = torch.cat((img_feat, grid_feat), dim=1)

        return img_feat, img_feat_mask


    def clevr_forward(self, feat_dict):
        grid_feat = feat_dict['GRID_FEAT']  # 196,1024

        img_feat_mask = make_mask(grid_feat)  # 61,1,1,196
        img_feat = self.grid_linear(grid_feat)  # 196,512
        bs = img_feat.shape[0]
        grid_abs = self.grid_abs(img_feat.view(bs, 14, 14, -1))  # (bs, num, dim)
        #grid_abs = self.abs_position_embedding(img_feat.view(bs, 14, 14, -1))  # (bs, num, dim)

        return img_feat, img_feat_mask, grid_abs



