import argparse
import time
import os

import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

import test_yolov3 as test  # Import test.py to get mAP after each epoch
from utils.parse_config import parse_data_cfg
from module.parse_model import *
import utils.torch_utils as torch_utils
from utils.utils import *
from module.models import *
from data.player_dataset import *
import pdb

#Original
hyp = {'lr0': 0.0001,  # initial learning rate
        'lrf': -5.,  # final learning rate = lr0 * (10 ** lrf)
        'momentum': 0.9,  # SGD momentum
        'weight_decay': 0.0005}  # optimizer weight decay


def train(
        cfg,
        data_cfg,
        resume=False,
        epochs=273,  # 500200 batches at bs 64, dataset length 117263
        batch_size=16,
        accumulate=1,
        weights_path='weights',
        init_weights='yolov3-player_stage2_start.81'
):
    #init_seeds()
    weights = weights_path + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device, n_gpu = torch_utils.select_device()

    #Image size
    cfg_model = parse_cfg(cfg)
    img_size = (int(cfg_model[0]['width']), int(cfg_model[0]['height']))
    
    # Configure run
    train_path = parse_data_cfg(data_cfg)['train_path']
    train_set = parse_data_cfg(data_cfg)['train_set']

    
    # Initialize model
    model = Darknet(cfg).to(device)

    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    
    if resume:  # Load previously saved model(resume from latest.pt)
        chkpt = torch.load(latest, map_location=device)  # load checkpoint
        model.load_state_dict(chkpt['model'])

        start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_loss = chkpt['best_loss']
        del chkpt

    else:  # Initialize model with backbone (optional)
        model.load_weights(weights + init_weights)

    # Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 440, 1097], gamma=0.1, last_epoch=start_epoch-1)
    
    # Dataset
    dataset = YoloDataSets(data_path=train_path, 
                            input_size=img_size, 
                            batch_size=batch_size, 
                            image_set=train_set, 
                            augment=True, jitter_x=0.3, jitter_y=0.3)
    
    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend=opt.backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
        model = torch.nn.parallel.DistributedDataParallel(model)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=False,  # disable rectangular training if True
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)
    

    # Mixed precision training https://github.com/NVIDIA/apex
    # install help: https://github.com/NVIDIA/apex/issues/259
    mixed_precision = False
    if mixed_precision:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Start training
    t = time.time()
    model.hyp = hyp  # attach hyperparameters to model
    #model_info(model)
    
    nb = len(dataloader)
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    
    n_burnin = int(cfg_model[0]["burn_in"])   # burn-in batches

    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) % ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler
        scheduler.step(epoch)

        mloss = torch.zeros(5).to(device)  # mean losses
        for i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            nt = len(targets)
            #plot_images(imgs=imgs, targets=targets, fname='train_batch%d.jpg' % i)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr
            

            if i == 0:
                print('learning rate: %g' % optimizer.param_groups[0]['lr'])
            # Run model
            pred, loss, loss_items = model(imgs, targets)
            loss = torch.mean(loss)
            n_ = int(loss_items.size()[0] / 5)
            loss_items = torch.mean(loss_items.view((n_, 5)), 0)



            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            # Update running mean of tracked metrics
            mloss = (mloss * i + loss_items) / (i + 1)
            # Print batch results
            s = ('%8s%12s' + '%10.3g' * 7) % ('%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, nt, time.time() - t)
 
            t = time.time()
            print(s)

        
        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 5)) or epoch == epochs - 1:
            with torch.no_grad():
                results, maps = test.test(cfg, data_cfg, batch_size=batch_size, img_size=img_size, model=model,
                                    conf_thres=0.1, iou_thres=0.4)
        
        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best loss
        test_loss = results[4]
        if test_loss < best_loss:
            best_loss = test_loss
        
        # Save training results
        save = True and not opt.nosave
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            
            # Save best checkpoint
            if best_loss == test_loss:
                torch.save(chkpt, best)
            
            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt
    
    return results


def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))
    with open('evolve.txt', 'a') as f:
        f.write(c + b + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=600, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--multi-scale', action='store_true', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=416, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--transfer', action='store_true', help='transfer learning flag')
    parser.add_argument('--num-workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str, help='distributed training init method')
    parser.add_argument('--rank', default=0, type=int, help='distributed training node rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--nosave', action='store_true', help='do not save training results')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='run hyperparameter evolution')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    parser.add_argument('--init_weights', type=str, default='yolov3-player_stage2_start.81', help='initialization model weights file name')
    parser.add_argument('--weights', type=str, default='weights', help='weights file path')

    opt = parser.parse_args()
    print(opt, end='\n\n')

    if opt.evolve:
        opt.notest = True  # save time by only testing final epoch
        opt.nosave = True  # do not save checkpoints

    # Train
    results = train(
        opt.cfg,
        opt.data_cfg,
        resume=opt.resume or opt.transfer,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulate=opt.accumulate,
        weights_path=opt.weights,
        init_weights=opt.init_weights,
    )

    # Evolve hyperparameters (optional)
    if opt.evolve:
        best_fitness = results[2]  # use mAP for fitness

        # Write mutation results
        print_mutation(hyp, results)

        gen = 50  # generations to evolve
        for _ in range(gen):

            # Mutate hyperparameters
            old_hyp = hyp.copy()
            torch_utils.init_seeds(seed=int(time.time()))
            s = [.2, .2, .2, .2, .2, .3, .2, .2, .02, .3]
            for i, k in enumerate(hyp.keys()):
                x = (np.random.randn(1) * s[i] + 1) ** 1.1  # plt.hist(x.ravel(), 100)
                hyp[k] = hyp[k] * float(x)  # vary by about 30% 1sigma

            # Clip to limits
            keys = ['iou_t', 'momentum', 'weight_decay']
            limits = [(0, 0.90), (0.80, 0.95), (0, 0.01)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Normalize loss components (sum to 1)
            keys = ['xy', 'wh', 'cls', 'conf']
            s = sum([v for k, v in hyp.items() if k in keys])
            for k in keys:
                hyp[k] /= s

            # Determine mutation fitness
            results = train(
                opt.cfg,
                opt.data_cfg,
                img_size=opt.img_size,
                resume=opt.resume or opt.transfer,
                transfer=opt.transfer,
                epochs=opt.epochs,
                batch_size=opt.batch_size,
                accumulate=opt.accumulate,
                multi_scale=opt.multi_scale,
            )
            mutation_fitness = results[2]

            # Write mutation results
            print_mutation(hyp, results)

            # Update hyperparameters if fitness improved
            if mutation_fitness > best_fitness:
                # Fitness improved!
                print('Fitness improved!')
                best_fitness = mutation_fitness
            else:
                hyp = old_hyp.copy()  # reset hyp to

            # # Plot results
            # import numpy as np
            # import matplotlib.pyplot as plt
            #
            # a = np.loadtxt('evolve.txt')
            # x = a[:, 3]
            # fig = plt.figure(figsize=(14, 7))
            # for i in range(1, 10):
            #     plt.subplot(2, 5, i)
            #
