import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init

'''
bbox需要分割得到x和y吗？
要是分割，后面再拼接到一起，不是一样的效果吗？

'''

def absolutePostion(bbox, img_wh):
    # bbox [64,100,4]  img_wh [64,2] num_r = 100
    # xmin, ymin, xmax, ymax = bbox[:, :, 0], bbox[:, :, 1], bbox[:, :, 2], bbox[:, :, 3]
    # xmin, ymin, xmax, ymax = torch.split(bbox, [1, 1, 1, 1], dim=2)
    cx = (bbox[:, :, 2] + bbox[:, :, 0]) * 0.5
    cy = (bbox[:, :, 1] + bbox[:, :, 3]) * 0.5
    w = (bbox[:, :, 2] - bbox[:, :, 0]) + 1.
    h = (bbox[:, :, 3] - bbox[:, :, 1]) + 1.
    area = w * h
    # print(area.shape)

    expand_wh = torch.cat([img_wh, img_wh], dim=1).unsqueeze(dim=1)  # (bs, 1, 4)
    bbox = torch.stack([cx, cy, w, h], dim=2)
    bbox = bbox / expand_wh  # (bs, num_r, 4)
    ratio_area = area / (img_wh[:, 0] * img_wh[:, 1]).unsqueeze(-1)  # (bs, num_r)
    ratio_area = ratio_area.unsqueeze(-1)   # (bs, num_r, 1)
    # S = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])/(img_wh[0] * img_wh[1])
    res = torch.cat([bbox, ratio_area], dim=-1)  # 64,100,5
    # print(res.shape)

    return res


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w).cuda()
        j = torch.arange(h).cuda()
        x_emb = self.col_embed(i).cuda()
        y_emb = self.row_embed(j).cuda()
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        not_mask = (mask == False)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # .permute(0, 3, 1, 2)
        pos = pos.flatten(1, 2)
        return pos

def relativePostion(bbox, dim_g=64, wave_len=1000):
    # bbox [64,100,4]  img_wh [64,2] num_r = 100
    batch_size = bbox.size(0)
    x_min, y_min, x_max, y_max = torch.chunk(bbox, 4, dim=-1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    wh = w / h
    area = w * h

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))
    # print(res.shape)

    size = delta_h.size()
    delta_x = delta_x.view(batch_size, size[1], size[2], 1)
    delta_y = delta_y.view(batch_size, size[1], size[2], 1)
    delta_w = delta_w.view(batch_size, size[1], size[2], 1)
    delta_h = delta_h.view(batch_size, size[1], size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    feat_range = torch.arange(dim_g / 8).cuda()
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], size[2], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(size[0], size[1], size[2], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1)

    return embedding  # 64,100,100,64
