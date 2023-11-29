# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.RTDETRIns.pre_decoder')
def rtdetrins__pre_decoder__default(
    self, memory: Tensor, memory_mask: Tensor, mask_features: Tensor,
    spatial_shapes: Tensor, batch_data_samples = None) -> Tuple[Dict, Dict]:
    from mmdet.models.layers.transformer import inverse_sigmoid
    from mmdet.models.detectors.rtdetr_ins import masks_to_boxes
    from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

    bs, _, c = memory.shape
    num_queries = self.test_cfg.get('max_per_img', self.num_queries)

    output_memory = memory * self.output_proposals_valid
    output_memory = self.memory_trans_fc(output_memory)
    output_memory = self.memory_trans_norm(output_memory)

    enc_outputs_class = self.bbox_head.cls_branches[
        self.decoder.num_layers](output_memory)
    topk_indices = torch.topk(
        enc_outputs_class.max(-1)[0], k=num_queries, dim=1)[1]

    query = torch.gather(
        output_memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, c))

    enc_outputs_mask = self.bbox_head.mask_branches[
        self.decoder.num_layers](query)  # shuold norm?
    topk_mask = torch.einsum(
        "bqc,bchw->bqhw", enc_outputs_mask, mask_features)

    # unified reference points
    h, w = topk_mask.shape[-2:]
    factor = topk_mask.new_tensor([w, h, w, h]).unsqueeze(0)
    masks = topk_mask.detach().reshape(-1, h, w) > 0
    topk_coords_xyxy = masks_to_boxes(masks).reshape(bs, -1, 4)
    topk_coords_normalized = bbox_xyxy_to_cxcywh(topk_coords_xyxy) / factor
    topk_coords_unact = inverse_sigmoid(topk_coords_normalized)

    decoder_inputs_dict = dict(
        query=query,
        memory=memory,
        reference_points=topk_coords_unact,
        dn_mask=None)
    head_inputs_dict = dict(mask_features=mask_features)
    return decoder_inputs_dict, head_inputs_dict


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.RTDETRInsHead.forward')
def rtdetrinshead__forward__default(self, hidden_states: Tensor,
    references: List[Tensor], mask_features: Tensor) -> Tuple[Tensor, Tensor]:
    outputs_class = self.cls_branches[-2](hidden_states)
    outputs_coord = references.sigmoid()
    tmp_mask_preds = self.mask_branches[-2](hidden_states)
    outputs_mask = torch.einsum(
        "bqc,bchw->bqhw", tmp_mask_preds, mask_features)
    return [outputs_class], [outputs_coord], [outputs_mask]


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.dense_heads.RTDETRInsHead.predict_by_feat')
def rtdetrinshead__predict_by_feat__default(
    self, all_cls_scores_list: List[Tensor],
    all_bbox_preds_list: List[Tensor], all_layers_mask_preds: Tensor,
    batch_img_metas: List[dict], rescale: bool = True):
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

    cls_scores = all_cls_scores_list[-1]
    bbox_preds = all_bbox_preds_list[-1]
    mask_preds = all_layers_mask_preds[-1]
    img_shape = batch_img_metas[0]['img_shape']
    if isinstance(img_shape, list):
        img_shape = torch.tensor(
            img_shape, dtype=torch.long, device=cls_scores.device)
    img_shape = img_shape.unsqueeze(0)

    max_per_img = self.test_cfg.get('max_per_img', len(cls_scores[0]))
    batch_size = cls_scores.size(0)
    # `batch_index_offset` is used for the gather of concatenated tensor

    # supports dynamical batch inference
    if self.loss_cls.use_sigmoid:
        batch_index_offset = torch.arange(batch_size).to(
            cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(
            batch_size, max_per_img)
        cls_scores = cls_scores.sigmoid()
        scores, indexes = cls_scores.flatten(1).topk(max_per_img, dim=1)
        det_labels = indexes % self.num_classes
        bbox_index = indexes // self.num_classes
        bbox_index = (bbox_index + batch_index_offset).view(-1)
        bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
        bbox_preds = bbox_preds.view(batch_size, -1, 4)
        mask_preds_size = (mask_preds.size(-2), mask_preds.size(-1))
        mask_preds = mask_preds.view(-1, *mask_preds_size)[bbox_index]
        mask_preds = mask_preds.view(batch_size, -1, *mask_preds_size)
    else:
        scores, det_labels = F.softmax(cls_scores, dim=-1)[..., :-1].max(-1)
        scores, bbox_index = scores.topk(max_per_img, dim=1)
        batch_inds = torch.arange(
            batch_size, device=scores.device).unsqueeze(-1)
        bbox_preds = bbox_preds[batch_inds, bbox_index, ...]
        mask_preds = mask_preds[batch_inds, bbox_index, ...]
        # add unsqueeze to support tensorrt
        det_labels = det_labels.unsqueeze(-1)[batch_inds, bbox_index,
                                              ...].squeeze(-1)

    det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
    det_bboxes.clamp_(min=0., max=1.)
    shape_scale = img_shape.flip(1).repeat(1, 2).unsqueeze(1)
    det_bboxes = det_bboxes * shape_scale
    det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)
    masks = F.interpolate(
        mask_preds,
        size=img_shape.squeeze(0).tolist(),
        mode='bilinear',
        align_corners=False)
    masks = masks.sigmoid() > 0.5
    return det_bboxes, det_labels, masks
