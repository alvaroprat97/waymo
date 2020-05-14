import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import WaymoDataset, Resizer, Normalizer, Augmenter, collater
from src.model import EfficientDet
# from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.notebook import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--cam_view", type=str, default="FRONT", help="Camera view to train on [FRONT, SIDE_LEFT...]")
    parser.add_argument("--log_path", type=str, default="tensorboard/efficientdet_waymo")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--train_depth", type=int, default=2)
    parser.add_argument("--pretrained_model", type=str, default='trained_models/signatrix_efficientdet_coco.pth', help="Path of pretrained model")
    parser.add_argument("--backbone", type=str, default="efficientnet-b7")
    parser.add_argument("-f", type=str, default="bam")

    args = parser.parse_args()
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def transfer_learning(model, train_depth=2):
    mlist = next(m for m in model.children() if isinstance(m, torch.nn.modules.container.ModuleList))
    print(f"Num trainable params before:{count_parameters(mlist)}")
    layers_to_freeze = mlist[:-train_depth]
    for l in layers_to_freeze:
        freeze_params(l)
    print(f"Num trainable params after:{count_parameters(mlist)}")
    print(f"Layers 0 to {len(layers_to_freeze)-1} frozen, only training on last {train_depth} layers")

def freeze_params(l):
    for p in l.parameters():
        if p.requires_grad:
            p.requires_grad = False

def train(opt):
    num_gpus = 1
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    training_params = {"batch_size": opt.batch_size * num_gpus,
                       "shuffle": True,
                       "drop_last": True,
                       "collate_fn": collater,
                       "num_workers": 12}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": collater,
                   "num_workers": 12}

    training_set = WaymoDataset(
        cameras=[opt.cam_view],scope='training',
        transform=transforms.Compose([Normalizer(), Resizer()]))
    training_generator = DataLoader(training_set, **training_params)

    test_set = WaymoDataset(
        cameras=[opt.cam_view], scope='validation',
        transform=transforms.Compose([Normalizer(), Resizer()]))
    test_generator = DataLoader(test_set, **test_params)


    if opt.pretrained_model:
        print(f'Using pretrained model from {opt.pretrained_model}')
        if torch.cuda.is_available():
            model = torch.load(opt.pretrained_model).module
        else:
            model = torch.load(
                opt.pretrained_model,
                map_location=torch.device('cpu')).module

        transfer_learning(model.backbone_net.model, opt.train_depth)
        num_classes = training_set.num_classes()
        model.classifier.num_class = 4
        model.header = torch.nn.Conv2d(
            model.num_channels,
            model.classifier.num_anchors * model.classifier.num_classes,
            kernel_size=3, stride=1, padding=1)


    else:
        model = EfficientDet(num_classes=4)


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    # writer = SummaryWriter(opt.log_path)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    best_loss = 1e5
    best_epoch = 0
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        model.train()
        # if torch.cuda.is_available():
        #     model.module.freeze_bn()
        # else:
        #     model.freeze_bn()
        epoch_loss = []
        progress_bar = tqdm(training_generator)
        for iter, data in enumerate(progress_bar):
#             try:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
            else:
                cls_loss, reg_loss = model([data['img'].float(), data['annot']])

            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            loss = cls_loss + reg_loss
            if loss == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss.append(float(loss))
            total_loss = np.mean(epoch_loss)
            if iter % 5 == 0:
                print(f'Total loss at iteration {iter}: {total_loss}')
            progress_bar.set_description(
                'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                    epoch + 1, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss, reg_loss, loss,
                    total_loss))
            # writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
            # writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
            # writer.add_scalar('Train/Classfication_loss (focal loss)', cls_loss, epoch * num_iter_per_epoch + iter)
            # Save every 100 samples
            if iter % 200 ==0:
                print(f"Saving model at :{opt.saved_path}/efficientdet_waymo.pth")
                torch.save(model, os.path.join(opt.saved_path, "efficientdet_waymo.pth"))

#             except Exception as e:
#                 continue
        scheduler.step(np.mean(epoch_loss))

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(test_generator):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                    else:
                        cls_loss, reg_loss = model([data['img'].float(), data['annot']])

                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss_classification_ls.append(float(cls_loss))
                    loss_regression_ls.append(float(reg_loss))

            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss

            print(
                'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    epoch + 1, opt.num_epochs, cls_loss, reg_loss,
                    np.mean(loss)))
            # writer.add_scalar('Test/Total_loss', loss, epoch)
            # writer.add_scalar('Test/Regression_loss', reg_loss, epoch)
            # writer.add_scalar('Test/Classfication_loss (focal loss)', cls_loss, epoch)

            if loss + opt.es_min_delta < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(model, os.path.join(opt.saved_path, "efficientdet_waymo.pth"))

                dummy_input = torch.rand(opt.batch_size, 3, 512, 512)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                if isinstance(model, nn.DataParallel):
                    model.module.backbone_net.model.set_swish(memory_efficient=False)

                    torch.onnx.export(model.module, dummy_input,
                                      os.path.join(opt.saved_path, "efficientdet_waymo.onnx"),
                                      verbose=False)
                    model.module.backbone_net.model.set_swish(memory_efficient=True)
                else:
                    model.backbone_net.model.set_swish(memory_efficient=False)

                    torch.onnx.export(model, dummy_input,
                                      os.path.join(opt.saved_path, "efficientdet_waymo.onnx"),
                                      verbose=False)
                    model.backbone_net.model.set_swish(memory_efficient=True)

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                break
    # writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
