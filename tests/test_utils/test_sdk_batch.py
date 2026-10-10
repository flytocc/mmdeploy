from mmengine import Config

from mmdeploy.backend.sdk.export_info import _get_model_batch_size


def test_get_model_batch_size_from_tensorrt_profile():
    deploy_cfg = Config(dict(
        backend_config=dict(
            type="tensorrt",
            model_inputs=[
                dict(input_shapes=dict(input=dict(max_shape=[8, 3, 640, 640])))
            ])))
    assert _get_model_batch_size(deploy_cfg) == 8


def test_get_model_batch_size_uses_smallest_profile_limit():
    deploy_cfg = Config(dict(
        backend_config=dict(
            type="tensorrt",
            model_inputs=[
                dict(input_shapes=dict(
                    image=dict(max_shape=[8, 3, 640, 640]),
                    mask=dict(max_shape=[4, 1, 640, 640])))
            ])))
    assert _get_model_batch_size(deploy_cfg) == 4


def test_get_model_batch_size_defaults_to_one():
    deploy_cfg = Config(dict(
        backend_config=dict(type="tensorrt", model_inputs=[])))
    assert _get_model_batch_size(deploy_cfg) == 1
