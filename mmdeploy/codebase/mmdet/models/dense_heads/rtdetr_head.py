# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.RTDETR.pre_decoder')
def rtdetr__pre_decoder__default(
    self, memory: Tensor, memory_mask: Tensor, spatial_shapes: Tensor,
    batch_data_samples = None) -> Tuple[Dict, Dict]:
    """Rewrite `pre_decoder` of `RTDETR` for default backend."""
    bs, _, c = memory.shape

    output_memory, output_proposals = self.gen_encoder_output_proposals(
        memory, memory_mask, spatial_shapes)

    enc_outputs_class = self.bbox_head.cls_branches[
        self.decoder.num_layers](output_memory)
    topk_indices = torch.topk(
        enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

    query = torch.gather(
        output_memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, c))
    topk_proposals = torch.gather(
        output_proposals, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))

    topk_coords_unact = self.bbox_head.reg_branches[
        self.decoder.num_layers](query) + topk_proposals

    decoder_inputs_dict = dict(
        query=query,
        memory=memory,
        reference_points=topk_coords_unact,
        dn_mask=None)
    head_inputs_dict = dict()
    return decoder_inputs_dict, head_inputs_dict


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.RTDETR.forward_decoder')
def rtdetr__forward_decoder__default(
    self, query: Tensor, memory: Tensor, memory_mask: Tensor,
    reference_points: Tensor, spatial_shapes: Tensor,
    level_start_index: Tensor, valid_ratios: Tensor,
    dn_mask: Optional[Tensor] = None, **kwargs) -> Dict:
    """Rewrite `forward_decoder` of `RTDETR` for default backend."""
    inter_states, references = self.decoder(
        query=query,
        value=memory,
        key_padding_mask=memory_mask,
        self_attn_mask=dn_mask,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        reg_branches=self.bbox_head.reg_branches,
        **kwargs)
    return dict(hidden_states=inter_states, references=references)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.layers.RTDETRTransformerDecoder.forward')
def rtdetrtransformerdecoder__forward__default(
    self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
    self_attn_mask: Tensor, reference_points: Tensor,
    spatial_shapes: Tensor, level_start_index: Tensor,
    valid_ratios: Tensor, reg_branches: nn.ModuleList,
    **kwargs) -> Tuple[Tensor, Tensor]:
    """Rewrite `forward` of `RTDETRTransformerDecoder` for default backend."""
    reference_points_unact, reference_points = \
        reference_points, reference_points.sigmoid()

    for lid, layer in enumerate(self.layers):
        reference_points_input = reference_points[:, :, None]
        query_pos = self.ref_point_head(reference_points)

        query = layer(
            query,
            query_pos=query_pos,
            value=value,
            key_padding_mask=key_padding_mask,
            self_attn_mask=self_attn_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reference_points=reference_points_input,
            **kwargs)

        if reg_branches is not None:
            tmp = reg_branches[lid](query)
            assert reference_points.shape[-1] == 4
            reference_points_unact = tmp + reference_points_unact
            reference_points = reference_points_unact.sigmoid()

    return query, reference_points


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.RTDETRHead.forward')
def rtdetrhead__forward__default(
    self, hidden_states: Tensor, references: Tensor) -> Tuple[Tensor, Tensor]:
    """Rewrite `forward` of `RTDETRHead` for default backend."""
    outputs_class = self.cls_branches[-2](hidden_states)
    return [outputs_class], [references]
