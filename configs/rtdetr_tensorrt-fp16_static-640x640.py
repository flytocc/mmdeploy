_base_ = ['./rtdetr_tensorrt_static-640x640.py']
backend_config = dict(common_config=dict(fp16_mode=True))
