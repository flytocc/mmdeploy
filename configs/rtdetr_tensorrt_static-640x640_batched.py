_base_ = ['./rtdetr_tensorrt_static-640x640.py']
onnx_config = dict(
    dynamic_axes={
        'input': {
            0: 'batch',
        },
        'dets': {
            0: 'batch',
        },
        'labels': {
            0: 'batch',
        },
    }, )
backend_config = dict(
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[4, 3, 640, 640],
                    max_shape=[8, 3, 640, 640])))
    ])
