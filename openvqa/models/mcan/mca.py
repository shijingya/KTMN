# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ------------------------------
# ----  问题文本CONTEXT Multi-Head Attention ----
# ------------------------------

class C_MHAtt(nn.Module):
    def __init__(self, __C):
        super(C_MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.lin_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.avgpool_c = nn.AdaptiveAvgPool2d((1, None))
        self.lin_c = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        # self.lin_c = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.lin_ac = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_cc = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.linear_cp = nn.Linear(__C.HIDDEN_SIZE, 1)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, s):
        n_batches = q.size(0)

        g_k = self.lin_ac(self.avgpool_c(s))  # B,1,512
        merge_p = self.lin_c(s) + self.linear_cc(g_k)  # B,N,512
        context_p = torch.sigmoid(merge_p)
        context_pp = self.linear_cp(context_p)  # B,N,1
        context_gp = torch.sigmoid(context_pp)  # B,N,1

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(  # B,N,512
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        atted = context_gp * atted + atted  # B,N,512
        # atted = self.lin_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(  # 64,8,14,14
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ------------------------------------------------
# ---- 问题引导的图像上下文、绝对位置嵌入  ----
# -----------------------------------------------

class context_IMHAtt(nn.Module):
    def __init__(self, __C):
        super(context_IMHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.avgpool_g = nn.AdaptiveAvgPool2d((1, None))
        self.linear_mc = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_ic = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_icc = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.linear_icp = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.lin_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.linear_c = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.linear_c1 = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_c2 = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k1 = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q1 = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.lin_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.lin_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.g_k = nn.Linear(1, 1)
        self.g_q = nn.Linear(1, 1)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

# q,k,v--[64,100,512] (batch,N,512) region_try----[64,100,512]  c[64,1,512]
    def forward(self, v, k, q, mask, c, region_try):
        n_batches = q.size(0)

        v = self.linear_v(v).view(  # 64,8,100,64
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

# image global representation
        #g_i = self.linear_mc(torch.mean(q, dim=1, keepdim=True))  # B,1,512  图像区域特征
        g_i = self.linear_mc(self.avgpool_g(q))  # B,1,512  图像网格特征
        merge_pi = self.linear_ic(q) + self.linear_icc(g_i)  # B,N,512
        # context_pi = torch.sigmoid(merge_pi)
        context_pi = torch.sigmoid(merge_pi)
        context_ppi = self.linear_icp(context_pi)  # B,N,1
        context_gpi = torch.sigmoid(context_ppi)  # B,N,1

# 对q和k进行优化
    # 第一种：k = k*sigmoid(w1c+w2k)+region, q = q*sigmoid(w1c+w3q)+region   71.13
        # cc = self.linear_c(c)  # 64,1,1
        #alpha = torch.sigmoid(cc + self.lin_q(q))  # 64,100,1
        #betla = torch.sigmoid(cc + self.lin_k(k))
        #q = q * alpha
        #k = k * betla
        # q = q + region_try
        # k = k + region_try

    # 第二种：k = （k+region）*sigmoid(w1c+w2k), q = (q+region)*sigmoid(w1c+w2q) ×
        # alpha = torch.sigmoid(cc + self.lin_q(q))
        # betla = torch.sigmoid(cc + self.lin_k(k))
        # q = (q + region_try) * alpha
        # k = (k + region_try) * betla

    # 第三种
        # alpha = self.g_q(context_gpi)
        # betla = self.g_k(context_gpi)
        # q = q * alpha
        # k = k * betla

    # 第四种   ×
        '''
        c1 = torch.sigmoid(self.linear_c1(c)) # 64,1,1
        c2 = torch.sigmoid(self.linear_c2(c))
        k = (1 + c1) * self.linear_k1(k)
        q = (1 + c2) * self.linear_q1(q)
        k = k + region_try
        q = q + region_try
        '''
        # ck = torch.sigmoid(self.linear_c1(c))  # 64,1,512
        # cq = torch.sigmoid(self.linear_c2(c))
        # k = k * ck + region_try
        # q = q * cq + region_try

    # 第五种 q = q*sigmoid[w3(w1c+q)]+region  w1[512,512],w3[512,1]
    #     ck = self.linear_c1(c) + k  # 64,1,512 + 64,100,512
    #     cq = self.linear_c2(c) + q
    #
    #     alpha = torch.sigmoid(self.lin_k(ck))  # 64,100,1
    #     betla = torch.sigmoid(self.lin_q(cq))
    #
    #     k = k * alpha
    #     q = q * betla
    #
    #     k = k + region_try
    #     q = q + region_try

    # 第六种：k = k*sigmoid(w1c+w2k)+region, q = q*sigmoid(w1c+w3q)+region  w1,w2,w3[512,512] 71.0  C2

        # cc = self.linear_c1(c)  # 64,1,512
        # alpha = torch.sigmoid(cc + self.linear_q1(q))  # 64,100,512
        # betla = torch.sigmoid(cc + self.linear_k1(k))
        # q = q * alpha
        # k = k * betla
        #
        # q = q + region_try
        # k = k + region_try

        # 第七种 q = q*sigmoid[w3(w1c+q)]+region  w1[512,512],w3[512,512]  71.08 √√√  C1!!!

        ck = self.linear_c1(c) + k  # 64,1,512 + 64,100,512
        cq = self.linear_c2(c) + q

        alpha = torch.sigmoid(self.lin_k(ck))  # 64,100,512
        betla = torch.sigmoid(self.lin_q(cq))

        k = k * alpha
        q = q * betla
        k = k + region_try
        q = q + region_try

        k = self.linear_k(k).view(  # [64,100,512]--->[64,8,100,64]
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(  # 64,8,100,64
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )
        atted = self.linear_merge(atted)  # 64,100,512

        atted = context_gpi * atted + atted

        atted = self.lin_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        # self.mhatt = MHAtt(__C)
        self.mhatt = C_MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        y = self.norm1(y + self.dropout1(
            # self.mhatt(y, y, y, y_mask)
            self.mhatt(y, y, y, y_mask, s=y)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = context_IMHAtt(__C)
        # self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask, c, region_try):
    # def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask, c=c, region_try=region_try)
            # self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, y, x, y_mask, x_mask, region_try):
    # def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # 获取文本上下文 c
        c = self.avgpool(y)  # [64,1,512]

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask, c, region_try)
            # x = dec(x, y, x_mask, y_mask)

        return y, x
