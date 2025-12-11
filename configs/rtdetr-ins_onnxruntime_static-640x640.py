_base_ = ['./rtdetr-ins_tensorrt_static-640x640.py']
_base_.backend_config = dict(type='onnxruntime')
