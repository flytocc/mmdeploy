# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Real
from typing import Optional, Tuple, Union

import torch
from packaging.version import InvalidVersion, Version
from torch.autograd import Function

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import (Backend, get_backend_version, get_ir_config,
                            get_root_logger)


def _is_tensorrt_bicubic_supported(ctx, input, size, scale_factor,
                                 align_corners):
    """Select the subset shared by native Resize and the legacy plugin.

    The plugin truncates scale factors when constructing output dimensions.
    It also computes align-corners ratios jointly for both spatial axes, so
    singleton outputs are excluded without specializing dynamic input shapes.
    """
    if (get_ir_config(ctx.cfg).get('opset_version', 11) < 11
            or input.dim() != 4 or input.dtype != torch.float32
            or size is not None):
        return False
    scales = ((scale_factor, scale_factor)
              if isinstance(scale_factor, Real) else scale_factor)
    if not isinstance(scales, (tuple, list)) or len(scales) != 2:
        return False
    minimum = 2 if align_corners else 1
    if not all(isinstance(s, Real) and math.isfinite(s)
               and s >= minimum and s == int(s)
               for s in scales):
        return False
    trt_version = get_backend_version().get('tensorrt')
    if trt_version is None:
        return False
    try:
        return Version(trt_version) >= Version('8.5.0')
    except InvalidVersion:
        return False


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.interpolate', backend='ncnn')
def interpolate__ncnn(input: torch.Tensor,
                      size: Optional[Union[int, Tuple[int], Tuple[int, int],
                                           Tuple[int, int, int]]] = None,
                      scale_factor: Optional[Union[float,
                                                   Tuple[float]]] = None,
                      mode: str = 'nearest',
                      align_corners: Optional[bool] = None,
                      recompute_scale_factor: Optional[bool] = None):
    """Rewrite `interpolate` for ncnn backend.

    ncnn require `size` should be constant in ONNX Node. We use `scale_factor`
    instead of `size` to avoid dynamic size.
    """
    ctx = FUNCTION_REWRITER.get_context()

    input_size = input.shape
    if scale_factor is None:
        scale_factor = [
            s_out / s_in for s_out, s_in in zip(size, input_size[2:])
        ]

    return ctx.origin_func(
        input,
        None,
        scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor)


@FUNCTION_REWRITER.register_rewriter(
    func_name='torch.nn.functional.interpolate', backend='rknn')
def interpolate__rknn(input: torch.Tensor,
                      size: Optional[Union[int, Tuple[int], Tuple[int, int],
                                           Tuple[int, int, int]]] = None,
                      scale_factor: Optional[Union[float,
                                                   Tuple[float]]] = None,
                      mode: str = 'nearest',
                      align_corners: Optional[bool] = None,
                      recompute_scale_factor: Optional[bool] = None):
    """Rewrite `interpolate` for rknn backend.

    rknn require `size` should be constant in ONNX Node. We use `scale_factor`
    instead of `size` to avoid dynamic size.
    """
    ctx = FUNCTION_REWRITER.get_context()
    input_size = input.shape
    if scale_factor is None:
        scale_factor = [(s_out / s_in)
                        for s_out, s_in in zip(size, input_size[2:])]
        if isinstance(scale_factor[0], torch.Tensor):
            scale_factor = [i.item() for i in scale_factor]

    return ctx.origin_func(
        input,
        None,
        scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor)


@FUNCTION_REWRITER.register_rewriter(
    'torch.nn.functional.interpolate',
    is_pytorch=True,
    backend=Backend.TENSORRT.value)
def interpolate__tensorrt(
    input: torch.Tensor,
    size: Optional[Union[int, Tuple[int], Tuple[int, int], Tuple[int, int,
                                                                 int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float]]] = None,
    mode: str = 'nearest',
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
):
    """Register default symbolic function for `interpolate`."""
    ctx = FUNCTION_REWRITER.get_context()

    if mode == 'bicubic' and _is_tensorrt_bicubic_supported(
            ctx, input, size, scale_factor, align_corners):
        return ctx.origin_func(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor)

    class BicubicInterpolate(Function):

        def __init__(self) -> None:
            super().__init__()

        @staticmethod
        def symbolic(g, input, scale_factor, align_corners):
            """Symbolic function for creating onnx op."""
            return g.op(
                'mmdeploy::TRTBicubicInterpolate',
                input,
                scale_factor_f=scale_factor,
                align_corners_i=align_corners)

        @staticmethod
        def forward(g, input, scale_factor, align_corners):
            """Run forward."""
            return ctx.origin_func(
                input,
                scale_factor=scale_factor,
                mode='bicubic',
                align_corners=align_corners)

    if 'bicubic' == mode:
        input_size = input.shape
        if isinstance(scale_factor, float):
            scale_factor = [scale_factor, scale_factor]
        if scale_factor is None:
            logger = get_root_logger()
            logger.warning(
                'ResizeLayer in TensorRT allow dynamic input shape with shape '
                'tensor. Which is not available for custom ops. Computed scale'
                '_factor might be the right way to get final shape.')
            scale_factor = [
                float(s_out / s_in)
                for s_out, s_in in zip(size, input_size[2:])
            ]
        return BicubicInterpolate.apply(input, scale_factor, align_corners)
    else:
        return ctx.origin_func(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor)
