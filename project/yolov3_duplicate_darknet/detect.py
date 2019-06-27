import argparse
import time
from sys import platform
import shutil
from pathlib import Path

from torch.utils.data import DataLoader

from utils.utils import *
from utils.parse_config import parse_data_cfg
from module.parse_model import *
import utils.torch_utils as torch_utils
from module.models import *
from data.player_dataset_detect import *


def detect(
        cfg,
        data_cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.15,
        nms_thres=0.5,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device, ngpu = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    #Image size
    cfg_model = parse_cfg(cfg)
    img_size = (int(cfg_model[0]['width']), int(cfg_model[0]['height']))

    # Initialize model
    model = Darknet(cfg).to(device)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        #_ = load_darknet_weights(model, weights)
        model.load_weights(weights)

    # Fuse Conv2d + BatchNorm2d layers
    #model.fuse()

    # Eval mode
    model.to(device).eval()

    # Configure run
    data_cfg = parse_data_cfg(data_cfg)
    nc = int(data_cfg['classes'])  # number of classes
    test_path = data_cfg['valid_path']  # path to test images
    test_set = data_cfg['valid_set']
    names = load_classes(data_cfg['names'])  # class names

    # Dataset
    batch_size = 1
    dataset = YoloDataSets(data_path=test_path, 
                            input_size=img_size, 
                            batch_size=batch_size, 
                            image_set=test_set, 
                            augment=False)
    

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)


    vid_path, vid_writer = None, None

    # Get classes and colors
    classes = load_classes(data_cfg['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    output_results = ""
    for batch_i, (path, imgs, im0, targets) in enumerate(tqdm(dataloader, desc='Computing mAP')):
        t = time.time()
        path = path[0]
        save_path = str(Path(output) / Path(path).name)
        
        im0 = im0.numpy()[0]
        imgs = imgs.to(device)
        targets = targets.to(device)
        _, _, height, width = imgs.shape

        # Run model
        inf_out = model(imgs, None)  # inference and training outputs

        # Run NMS
        det = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(imgs.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in det:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                    x = int(xyxy[0])
                    y = int(xyxy[1])
                    w = int(xyxy[2]) - x + 1
                    h = int(xyxy[3]) - y + 1
                    output_results = output_results + "{:s},{:d},{:d},{:d},{:d},{:f}\n".format(
                        Path(path).name, x, y, w,
                        h, conf)


                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('Done. (%.3fs)' % (time.time() - t))

        if save_images:  # Save generated image with detections
            cv2.imwrite(save_path, im0)
    with open("player_detection_results.txt", 'w') as pdrf:
        pdrf.write(output_results)

    if save_images and platform == 'darwin':  # macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            save_txt = True
        )
