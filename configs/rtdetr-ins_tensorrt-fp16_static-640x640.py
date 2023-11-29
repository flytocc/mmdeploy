_base_ = 'configs/rtdetr_tensorrt-fp16_static-640x640.py'

onnx_config = dict(output_names=['dets', 'labels', 'masks'])
codebase_config = dict(post_processing=dict(export_postprocess_mask=True))
