# coding: utf-8

conf_yolov3 = {
        'epoch': 100,
        'lr': 5e-4,
        'gamma': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'classes': 21,
        'ignore_threshold': 0.5,
        'img_size': 416,
        'batch_size': 32,
        'obj_scale': 1,
        'noobj_scale': 0.5,
        'anchors': [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]]
}
