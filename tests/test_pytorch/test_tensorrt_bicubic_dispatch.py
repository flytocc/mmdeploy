# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import io
import tempfile
from types import SimpleNamespace

import onnx
import pytest
import torch
import torch.nn.functional as F
from mmengine import Config

from mmdeploy.core import RewriterContext


@pytest.fixture(scope='module', autouse=True)
def initialize_legacy_exporter():
    # PyTorch lazily populates the FX registry on its first legacy export.
    # Initialize it before RewriterContext snapshots that registry.
    torch.onnx.export(
        torch.nn.Identity(), torch.randn(1), io.BytesIO(), opset_version=17,
        dynamo=False)


class _Interpolate(torch.nn.Module):

    def __init__(self, size, scale_factor, align_corners):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=self.align_corners)


def _export_resize(monkeypatch, trt_version, opset, size, scale_factor,
                   align_corners, input_dtype=torch.float32):
    interpolate_module = importlib.import_module(
        'mmdeploy.pytorch.functions.interpolate')

    monkeypatch.setattr(interpolate_module, 'get_backend_version',
                        lambda: {'tensorrt': trt_version})
    cfg = Config({
        'backend_config': {
            'type': 'tensorrt'
        },
        'ir_config': {
            'opset_version': opset
        }
    })
    context = RewriterContext(cfg, 'tensorrt', opset=opset)
    model = _Interpolate(size, scale_factor, align_corners).eval()
    inputs = torch.randn(1, 2, 5, 7, dtype=input_dtype)
    with tempfile.NamedTemporaryFile(suffix='.onnx') as output_file:
        with context:
            torch.onnx.export(
                model,
                inputs,
                output_file.name,
                opset_version=opset,
                dynamo=False)
        return list(onnx.load(output_file.name).graph.node)


@pytest.mark.parametrize(
    'version,opset,size,scale,align,dtype,expected', [
        ('8.6.1', 11, None, (2.0, 3.0), False, torch.float32, 'Resize'),
        ('10.16.1.11', 17, None, (2.0, 3.0), False, torch.float32,
         'Resize'),
        ('10.16.1.11', 17, None, (2.0, 3.0), True, torch.float32, 'Resize'),
        ('8.4.3', 17, None, (2.0, 3.0), False, torch.float32,
         'TRTBicubicInterpolate'),
        ('10.16.1.11', 10, None, (2.0, 3.0), False, torch.float32,
         'TRTBicubicInterpolate'),
        ('10.16.1.11', 17, None, (1.7, 1.3), False, torch.float32,
         'TRTBicubicInterpolate'),
        ('10.16.1.11', 17, (7, 9), None, False, torch.float32,
         'TRTBicubicInterpolate'),
        ('10.16.1.11', 17, None, (2.0, 3.0), False, torch.float16,
         'TRTBicubicInterpolate'),
])
def test_tensorrt_bicubic_dispatch(monkeypatch, version, opset, size, scale,
                                   align, dtype, expected):
    nodes = _export_resize(monkeypatch, version, opset, size, scale, align,
                           dtype)
    resize_nodes = [node for node in nodes
                    if node.op_type in ('Resize', 'TRTBicubicInterpolate')]
    assert len(resize_nodes) == 1
    assert resize_nodes[0].op_type == expected
    assert resize_nodes[0].domain == ('' if expected == 'Resize' else
                                      'mmdeploy')
    if expected == 'Resize':
        attrs = {
            attr.name: onnx.helper.get_attribute_value(attr)
            for attr in resize_nodes[0].attribute
        }
        assert attrs['mode'] == b'cubic'
        assert attrs['cubic_coeff_a'] == pytest.approx(-0.75)
        assert attrs['coordinate_transformation_mode'] == (
            b'align_corners' if align else b'half_pixel')


@pytest.mark.parametrize('rank,dtype,scale,align,expected', [
    (3, torch.float32, (2.0, 2.0), False, False),
    (5, torch.float32, (2.0, 2.0), False, False),
    (4, torch.float16, (2.0, 2.0), False, False),
    (4, torch.float32, (1.0, 2.0), True, False),
    (4, torch.float32, (2.0, 2.0), True, True),
])
def test_tensorrt_bicubic_support_predicate(monkeypatch, rank, dtype, scale,
                                           align, expected):
    interpolate_module = importlib.import_module(
        'mmdeploy.pytorch.functions.interpolate')

    monkeypatch.setattr(interpolate_module, 'get_backend_version',
                        lambda: {'tensorrt': '10.16.1.11'})
    cfg = Config({
        'backend_config': {
            'type': 'tensorrt'
        },
        'ir_config': {
            'opset_version': 17
        }
    })
    context = SimpleNamespace(cfg=cfg)
    inputs = torch.randn(*([1] * rank), dtype=dtype)
    assert interpolate_module._is_tensorrt_bicubic_supported(
        context, inputs, None, scale, align) is expected
