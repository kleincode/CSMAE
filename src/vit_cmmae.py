# Adapted from https://github.com/facebookresearch/mae.
# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Tuple
import logging
from functools import partial

import torch
import torch.nn as nn
from src.pos_encoding import generate_2d_sincos_pos_embed
from timm.models.vision_transformer import Block, PatchEmbed, VisionTransformer

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from scipy.signal import wiener

import numpy as np


class CrossModalMaskedAutoencoderViT(VisionTransformer):
    """Masked Autoencoder with VisionTransformer backbone
    Adapted from https://github.com/facebookresearch/mae.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=24,
        pre_encoder_depth=12,
        shared_encoder_depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        global_pool="token",
        fc_norm=None,
        num_classes=0,
        norm_layer=nn.LayerNorm,
        masking_strategy="",
        modality_embedding=False,
        wiener_filter=False,
        identical_masking=True,
        disjoint_masking=False,
        **kwargs,
    ):
        assert in_chans == 12
        assert img_size % patch_size == 0

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=shared_encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            global_pool=global_pool,
            fc_norm=fc_norm,
            num_classes=num_classes,
            norm_layer=norm_layer,
            **kwargs,
        )

        in_chans_s1 = 2
        in_chans_s2 = 10
        self.pre_encoder_depth = pre_encoder_depth

        print(f"Architecture setup: {pre_encoder_depth}x separate encoders + {shared_encoder_depth}x shared encoders")


        # --------------------------------------------------------------------------
        # CMMAE S1 encoder specifics

        # we have to assign PatchEmbed to self.patch_embed since it's used parent ViT class,
        # otherwise these parameters do not receive gradients!
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans_s1, embed_dim) 
        self.cls_token = nn.parameter.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patches: int = self.patch_embed.num_patches # type: ignore

        self.modality_embedding = modality_embedding
        self.wiener_filter = wiener_filter

        self.identical_masking = identical_masking
        self.disjoint_masking = disjoint_masking

        # Legacy configuration of masking strategy        
        if masking_strategy:
            self.masking_strategy = masking_strategy
        else:
            if self.identical_masking:
                self.masking_strategy = 'identical'
            elif self.disjoint_masking:
                self.masking_strategy = 'disjoint'
            elif not self.identical_masking and not self.disjoint_masking:
                self.masking_strategy = 'random'
            else:
                assert False, "Invalid masking configuration."


        if pre_encoder_depth:
            self.blocks_s1 = nn.Sequential(
                *[
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for _ in range(pre_encoder_depth)
                ]
            )
            self.norm_s1 = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # CMMAE S2 encoder specifics
        self.patch_embed_s2 = PatchEmbed(img_size, patch_size, in_chans_s2, embed_dim)
        self.cls_token_s2 = nn.parameter.Parameter(torch.zeros(1, 1, embed_dim))
        assert self.patch_embed_s2.num_patches == self.num_patches

        if pre_encoder_depth:
            self.blocks_s2 = nn.Sequential(
                *[
                    Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for _ in range(pre_encoder_depth)
                ]
            )
            self.norm_s2 = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # S1 (S2) encoder specifics
        self.pos_embed = nn.parameter.Parameter(
                    torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
                )  # fixed sin-cos embedding

        # --------------------------------------------------------------------------
        # CMMAE shared encoder specifics
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(shared_encoder_depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------


        if self.modality_embedding:
            self.s1_embedding = nn.Linear(embed_dim, embed_dim)
            self.s2_embedding = nn.Linear(embed_dim, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = generate_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_s2.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.cls_token_s2, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x: torch.Tensor, mask_ratio: float, ids_shuffle: torch.Tensor, ids_restore: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder_s1(self, x: torch.Tensor, mask_ratio: float, ids_shuffle: torch.Tensor, ids_restore: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        if self.modality_embedding:
            x = self.s1_embedding(x)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, ids_shuffle, ids_restore)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if self.pre_encoder_depth:
            x = self.blocks_s1(x)
            x = self.norm_s1(x)

        return x, mask, ids_restore
    
    def forward_encoder_s2(self, x: torch.Tensor, mask_ratio: float, ids_shuffle: torch.Tensor, ids_restore: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # embed patches
        x = self.patch_embed_s2(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        if self.modality_embedding:
            x = self.s2_embedding(x)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, ids_shuffle, ids_restore)

        # append cls token
        cls_token = self.cls_token_s2 + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if self.pre_encoder_depth:
            x = self.blocks_s2(x)
            x = self.norm_s2(x)

        return x, mask, ids_restore
    
    def forward_shared_encoder(self, x):
        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0):

        N = imgs.shape[0]
        L = self.num_patches

        # sort noise for each sample
        noise = torch.rand(N, L, device=imgs.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        imgs_s1 = imgs[:, 10:, :, :]
        imgs_s2 = imgs[:, :10, :, :]

        if self.wiener_filter:
            np_list = np.asarray([wiener(x.cpu().detach().numpy(), 7) for x in imgs_s1[:, 0]])
            imgs_s1[:, 0] = torch.tensor(torch.from_numpy(np_list), device=imgs.device)
            np_list = np.asarray([wiener(x.cpu().detach().numpy(), 7) for x in imgs_s1[:, 1]])
            imgs_s1[:, 1] = torch.tensor(torch.from_numpy(np_list), device=imgs.device)

        # Pass S1 (S2) through corresponding encoder
        pre_feats_s1, mask_s1, ids_restore_s1 = self.forward_encoder_s1(imgs_s1, mask_ratio, ids_shuffle, ids_restore)

        if self.masking_strategy == 'identical':
            pass # identical masking does not alter the noise/shuffling values

        elif self.masking_strategy == 'disjoint':
            ids_shuffle = torch.flip(ids_shuffle, dims=[1])
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        elif self.masking_strategy == 'random':
            noise = torch.rand(N, L, device=imgs.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        else:
            assert False, 'Masking strategy not supported.'

        pre_feats_s2, mask_s2, ids_restore_s2 = self.forward_encoder_s2(imgs_s2, 1.-mask_ratio if self.masking_strategy == 'disjoint' else mask_ratio, ids_shuffle, ids_restore)

        # Pass obtained features through shared encoder
        feats_s1 = self.forward_shared_encoder(pre_feats_s1)
        feats_s2 = self.forward_shared_encoder(pre_feats_s2)

        # Classification head of transformer, either 
        # cls_token (default) or avg of all tokens
        out_s1 = self.forward_head(feats_s1)
        out_s2 = self.forward_head(feats_s2)

        if mask_ratio:
            return {
                's1': (out_s1, feats_s1, mask_s1, ids_restore_s1),
                's2': (out_s2, feats_s2, mask_s2, ids_restore_s2),
            }

        return {
            's1': out_s1,
            's2': out_s2,
        }


def vit_tiny(**kwargs):
    model = CrossModalMaskedAutoencoderViT(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_small(**kwargs):
    model = CrossModalMaskedAutoencoderViT(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base(**kwargs):
    model = CrossModalMaskedAutoencoderViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large(**kwargs):
    model = CrossModalMaskedAutoencoderViT(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge(**kwargs):
    if kwargs["patch_size"] != 14:
        logging.warning("Replaced patch size by 14 (default for MAE).")
        kwargs["patch_size"] = 14

    model = CrossModalMaskedAutoencoderViT(
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
