# Copyright (c) OpenMMLab. All rights reserved.
from packaging import version
from torch.onnx.symbolic_helper import parse_args

from mmdeploy.core import SYMBOLIC_REWRITER
from mmdeploy.utils import (Backend, get_backend, get_backend_version,
                            get_ir_config)


@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler(g,
                 input,
                 grid,
                 interpolation_mode,
                 padding_mode,
                 align_corners=False):
    """Symbolic function for `grid_sampler`.

    PyTorch does not support export grid_sampler to ONNX by default. We add the
    support here. `grid_sampler` will be exported as ONNX node
    'mmdeploy::grid_sampler'
    """
    return g.op(
        'mmdeploy::grid_sampler',
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners)


@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler_ppl(g,
                     input,
                     grid,
                     interpolation_mode,
                     padding_mode,
                     align_corners=False):
    """Symbolic function for `grid_sampler`.

    PyTorch does not support export grid_sampler to ONNX by default. We add the
    support here. `grid_sampler` will be exported as ONNX node
    'mmdeploy::grid_sampler'
    """
    return g.op(
        'mmcv::grid_sampler',
        input,
        grid,
        interpolation_mode_i=interpolation_mode,
        padding_mode_i=padding_mode,
        align_corners_i=align_corners)


@parse_args('v', 'v', 'i', 'i', 'i')
def grid_sampler_tensorrt(g,
                          input,
                          grid,
                          interpolation_mode,
                          padding_mode,
                          align_corners=False):
    """Export GridSample for TensorRT native support."""
    modes = {0: 'bilinear', 1: 'nearest', 2: 'bicubic'}
    padding_modes = {0: 'zeros', 1: 'border', 2: 'reflection'}
    return g.op(
        'GridSample',
        input,
        grid,
        mode_s=modes[interpolation_mode],
        padding_mode_s=padding_modes[padding_mode],
        align_corners_i=align_corners)


def _is_tensorrt_grid_sampler_supported(ctx, input):
    """Check whether TensorRT can consume the standard GridSample op."""
    opset = get_ir_config(ctx.cfg).get('opset_version', 11)
    if opset < 16:
        return False
    input_type = input.type()
    if not hasattr(input_type, 'dim') or input_type.dim() != 4:
        return False
    trt_version = get_backend_version().get('tensorrt')
    return trt_version is not None and version.parse(
        trt_version) >= version.parse('8.5.0')


@SYMBOLIC_REWRITER.register_symbolic('grid_sampler', is_pytorch=True)
def grid_sampler__default(*args):
    """Register default symbolic function for `grid_sampler`.

    Add support to grid_sample to ONNX.
    """
    ctx = SYMBOLIC_REWRITER.get_context()
    backend = get_backend(ctx.cfg)
    if backend == Backend.PPLNN:
        return grid_sampler_ppl(*args)
    if backend == Backend.TENSORRT and _is_tensorrt_grid_sampler_supported(
            ctx, args[1]):
        return grid_sampler_tensorrt(*args)
    else:
        return grid_sampler(*args)
