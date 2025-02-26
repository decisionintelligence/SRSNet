'''
* @author: EmpyreanMoon
*
* @create: 2025-02-26 16:27
*
* @description: 
'''
import torch.nn as nn
import torch
from ts_benchmark.baselines.srstst.layers.Embed import PositionalEmbedding
from einops import rearrange

class SRS(nn.Module):
    def __init__(self, d_model, patch_len, stride, seq_len, dropout, hidden_size):
        super(SRS, self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.padding = stride
        self.seq_len = seq_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))
        self.patch_num = (self.seq_len - self.patch_len + self.stride) // self.stride + 1

        self.scorer_select = nn.Sequential(nn.Linear(self.patch_len, hidden_size), nn.ReLU(),
                                           nn.Linear(hidden_size, self.patch_num))

        self.scorer_shuffle = nn.Sequential(nn.Linear(self.patch_len, hidden_size), nn.ReLU(),
                                            nn.Linear(hidden_size, 1))
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding_org = nn.Linear(patch_len, d_model, bias=False)
        self.value_embedding_rec = nn.Linear(patch_len, d_model, bias=False)
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Adaptive weight between Original View and Reconstruction View
        self.alpha = nn.Parameter(torch.tensor(2.0))

    def _origin_view(self, x):
        x_origin = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        origin_patches = rearrange(x_origin, 'b c n p -> (b c) n p')
        return origin_patches

    def _rec_view(self, x):
        x_rec = x.unfold(dimension=-1, size=self.patch_len, step=1)
        selected_patches = self._select(x_rec)
        shuffled_patches = self._shuffle(selected_patches)
        rec_patches = rearrange(shuffled_patches, 'b c n p -> (b c) n p')
        return rec_patches

    def _select(self, x_rec):
        scores = self.scorer_select(x_rec)
        indices = torch.argmax(scores, dim=-2, keepdim=True)
        max_scores = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv = (1 / max_scores[non_zero_mask]).detach()

        # [batch_size, n_vars, patch_num, patch_size]
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)

        # [batch_size, n_vars, patch_num, patch_size]
        max_scores[non_zero_mask] *= inv
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches

        return selected_patches

    def _shuffle(self, selected_patches):
        shuffle_scores = self.scorer_shuffle(selected_patches)
        shuffle_indices = torch.argsort(input=shuffle_scores, dim=-2)
        shuffled_scores = torch.gather(input=shuffle_scores, index=shuffle_indices, dim=-2)
        non_zero_mask = shuffled_scores != 0
        inv = (1 / shuffled_scores[non_zero_mask]).detach()

        shuffle_patch_indices = shuffle_indices.repeat(1, 1, 1, self.patch_len)
        shuffled_patches = torch.gather(input=selected_patches, index=shuffle_patch_indices, dim=-2)
        shuffled_scores[non_zero_mask] *= inv
        shuffled_patches = shuffled_scores * shuffled_patches

        return shuffled_patches

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        # padding for the original stride
        x = self.padding_patch_layer(x)

        rec_repr_space = self._rec_view(x)
        original_repr_space = self._origin_view(x)

        weight = torch.sigmoid(self.alpha)
        embedding = weight * self.value_embedding_org(original_repr_space) \
                    + (1 - weight) * self.value_embedding_rec(rec_repr_space) \
                    + self.position_embedding(original_repr_space)

        return self.dropout(embedding), n_vars