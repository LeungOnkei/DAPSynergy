# Copyright (c) 2026 [Leung]
# This code is provided for review purposes and will be finalized upon publication.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the original work is properly cited.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as m_gmp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from collections import Counter
import pandas as pd


class DepthwiseSeparableGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableGCNConv, self).__init__(aggr='add')  # "Add" aggregation.

        self.pointwise = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.propagate(edge_index, x=x, norm=norm)

        x = self.pointwise(x)

        return x

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Normalize node features.
        return norm.view(-1, 1) * x_j


class StructuralSelfAttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super(StructuralSelfAttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, 1)
        )

    def forward(self, x, batch):
        attn_weights = torch.sigmoid(self.attention(x))

        weighted_x = x * attn_weights

        return m_gmp(weighted_x, batch), attn_weights



class CrossModalAttentionPooling(nn.Module):
    def __init__(self, drug_channels, cell_channels, head=4):
        super(CrossModalAttentionPooling, self).__init__()
        self.head = head
        self.drug_channels = drug_channels
        self.cell_channels = cell_channels
        self.head_dim = drug_channels // head

        self.w_q = nn.Linear(cell_channels, drug_channels)
        self.w_k = nn.Linear(drug_channels, drug_channels)
        self.w_v = nn.Linear(drug_channels, drug_channels)

        self.fc_out = nn.Linear(drug_channels, drug_channels)

    def forward(self, drug_x, cell_x, batch):
        batch_size = cell_x.size(0)

        cell_expanded = cell_x[batch]

        Q = self.w_q(cell_expanded).view(-1, self.head, self.head_dim)
        K = self.w_k(drug_x).view(-1, self.head, self.head_dim)
        V = self.w_v(drug_x).view(-1, self.head, self.head_dim)

        scores = torch.einsum("qhd,khd->qh", Q, K) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=0).unsqueeze(-1)

        weighted = (attn * V).view(-1, self.head * self.head_dim)

        weighted = self.fc_out(weighted)

        return m_gmp(weighted, batch), attn


class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveFusion, self).__init__()
        self.alpha_generator = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Sigmoid()
        )

    def forward(self, struct_feat, cross_feat):
        combined = torch.cat([struct_feat, cross_feat], dim=1)

        alpha = self.alpha_generator(combined)

        return alpha * struct_feat + (1 - alpha) * cross_feat, alpha




class DAPSynergy_v2_multimodal(torch.nn.Module):
    def __init__(self, n_output=2, atom_dim=78, fp_dim=1024, physchem_dim=8, num_features_xt=954, output_dim=128,
                 dropout=0.2):
        super(DAPSynergy_v2_multimodal, self).__init__()

        self.n_output = n_output
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.drug1_conv1 = DepthwiseSeparableGCNConv(atom_dim, output_dim)
        self.drug1_conv2 = DepthwiseSeparableGCNConv(output_dim, output_dim * 2)

        self.drug2_conv1 = DepthwiseSeparableGCNConv(atom_dim, output_dim)
        self.drug2_conv2 = DepthwiseSeparableGCNConv(output_dim, output_dim * 2)

        self.fp_processor = nn.Sequential(
            nn.Linear(fp_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )

        self.physchem_processor = nn.Sequential(
            nn.Linear(physchem_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, output_dim // 4)
        )

        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim * 2)
        )

        self.drug1_struct_pool = StructuralSelfAttentionPooling(output_dim * 2)
        self.drug1_cross_pool = CrossModalAttentionPooling(output_dim * 2, output_dim * 2)
        self.drug1_fusion = AdaptiveFusion(output_dim * 2)

        self.drug2_struct_pool = StructuralSelfAttentionPooling(output_dim * 2)
        self.drug2_cross_pool = CrossModalAttentionPooling(output_dim * 2, output_dim * 2)
        self.drug2_fusion = AdaptiveFusion(output_dim * 2)

        self.drug1_feat_fusion = nn.Sequential(
            nn.Linear(output_dim * 2 + output_dim + output_dim // 4, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.drug2_feat_fusion = nn.Sequential(
            nn.Linear(output_dim * 2 + output_dim + output_dim // 4, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc1 = nn.Linear(output_dim * 6, 512)  # 2*(output_dim*2) + output_dim*2
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, n_output)

        self.bn1 = nn.BatchNorm1d(output_dim * 2)
        self.bn2 = nn.BatchNorm1d(output_dim * 6)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)



    def forward(self, data1, data2):

        x1_atom, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        fp1, physchem1 = data1.mixed_fp, data1.physchem_props

        x2_atom, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        fp2, physchem2 = data2.mixed_fp, data2.physchem_props

        cell_vector = self.reduction(cell)


        x1_atom = self.relu(self.drug1_conv1(x1_atom, edge_index1))
        x1_atom = self.dropout(x1_atom)
        x1_atom = self.relu(self.drug1_conv2(x1_atom, edge_index1))
        x1_atom = self.dropout(x1_atom)
        x1_atom = self.bn1(x1_atom)

        fp1 = fp1.view(-1, 1024)  # (batch_size, 1024)
        fp1_processed = self.fp_processor(fp1)

        physchem1 = physchem1.view(-1, 8)  # (batch_size, 8)
        physchem1_processed = self.physchem_processor(physchem1)

        struct_feat1, struct_attn1 = self.drug1_struct_pool(x1_atom, batch1)
        cross_feat1, cross_attn1 = self.drug1_cross_pool(x1_atom, cell_vector, batch1)
        drug1_gcn_feat, alpha1 = self.drug1_fusion(struct_feat1, cross_feat1)

        drug1_feat = torch.cat([drug1_gcn_feat, fp1_processed, physchem1_processed], dim=1)
        drug1_feat = self.drug1_feat_fusion(drug1_feat)

        x2_atom = self.relu(self.drug2_conv1(x2_atom, edge_index2))
        x2_atom = self.dropout(x2_atom)
        x2_atom = self.relu(self.drug2_conv2(x2_atom, edge_index2))
        x2_atom = self.dropout(x2_atom)
        x2_atom = self.bn1(x2_atom)


        fp2 = fp2.view(-1, 1024)  # (batch_size, 1024)
        fp2_processed = self.fp_processor(fp2)


        physchem2 = physchem2.view(-1, 8)  # (batch_size, 8)
        physchem2_processed = self.physchem_processor(physchem2)

        struct_feat2, struct_attn2 = self.drug2_struct_pool(x2_atom, batch2)
        cross_feat2, cross_attn2 = self.drug2_cross_pool(x2_atom, cell_vector, batch2)
        drug2_gcn_feat, alpha2 = self.drug2_fusion(struct_feat2, cross_feat2)

        drug2_feat = torch.cat([drug2_gcn_feat, fp2_processed, physchem2_processed], dim=1)
        drug2_feat = self.drug2_feat_fusion(drug2_feat)

        combined = torch.cat([drug1_feat, drug2_feat, cell_vector], dim=1)
        combined = self.bn2(combined)

        x = self.fc1(combined)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        out = self.out(x)

        return out, {
            'struct_attn1': struct_attn1,
            'cross_attn1': cross_attn1,
            'alpha1': alpha1,
            'struct_attn2': struct_attn2,
            'cross_attn2': cross_attn2,
            'alpha2': alpha2
        }

