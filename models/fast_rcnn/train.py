import os
import argparse

import torch.utils.data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import WaymoDataset, Resizer, Normalizer, Augmenter, collate_fn

# from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.notebook import tqdm
import functools


def get_args():
    parser = argparse.ArgumentParser(
        "Faster_RCNN")
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
    parser.add_argument("--log_path", type=str, default="tensorboard/fasterrcnn_waymo")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--freeze_layers", type=list, default=['backbone', 'rpn', 'roi_heads.box_head.fc6'])
    parser.add_argument("--pretrained_model", type=bool, default=True, help="Using pretrained model? True or False")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("-f", type=str, default="bam")

    args = parser.parse_args()
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def transfer_learning(model, layer_parts = ['backbone', 'rpn', 'roi_heads.box_head.fc6']):
    print(f"Num trainable params before:{count_parameters(model)}")
    layers_to_freeze = []
    text = ''
    for layer_part in layer_parts:
        layers_to_freeze.append(rgetattr(model, layer_part))
        text += layer_part + ', '
    for layer_group in layers_to_freeze:
        for layer in layer_group.modules():
            freeze_params(layer)
    print(f"Num trainable params after:{count_parameters(model)}")
    print(f"Layers: {text} are frozen.")

def freeze_params(layer):
    for parameter in layer.parameters():
        if parameter.requires_grad:
            parameter.requires_grad = False

def rgetattr(obj, attr):
    def _getattr(obj, attr):
        return getattr(obj, attr)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


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
                       "collate_fn": collate_fn,
                       "num_workers": 12}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                   "collate_fn": collate_fn,
                   "num_workers": 12}

    training_set = WaymoDataset(
        cameras=[opt.cam_view],scope='training',
        transform=transforms.Compose([Normalizer(), Resizer()]),
        mod='fast_rcnn')
    training_generator = DataLoader(training_set, **training_params)

    test_set = WaymoDataset(
        cameras=[opt.cam_view], scope='validation',
        transform=transforms.Compose([Normalizer(), Resizer()]),
        mod='fast_rcnn')
    test_generator = DataLoader(test_set, **test_params)
    
    
    print(f'Using pretrained model? {opt.pretrained_model}')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=opt.pretrained_model)
    # num_classes which is user-defined
    num_classes = training_set.num_classes()
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one, this will really need to be trained!
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('trained_models/fasterrcnn_resnet50_waymo.pth'))
    
    # only if we use the pretrained model
    if opt.pretrained_model:
        transfer_learning(model, opt.freeze_layers)

    # Chosing the device/cpu or gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    # writer = SummaryWriter(opt.log_path)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_loss = 1e5
    best_epoch = 0
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        model.train()
        epoch_loss = []
        progress_bar = tqdm(training_generator)
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()
            images = data[0]
            targets = data[1]
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]

            if torch.cuda.is_available():
                losses = model(images.cuda(), targets.cuda())
                cls_loss, reg_loss = losses['loss_classifier'], losses['loss_box_reg']
            else:
                losses = model(images, targets)
                cls_loss, reg_loss = losses['loss_classifier'], losses['loss_box_reg']

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
                print(f"Saving model at :{opt.saved_path}/fasterrcnn_resnet50_waymo.pth")
                torch.save(model.state_dict(), os.path.join(opt.saved_path, "fasterrcnn_resnet50_waymo.pth"))
#                 torch.save(model, os.path.join(opt.saved_path, "fasterrcnn_resnet50_waymo.pth"))

        scheduler.step(np.mean(epoch_loss))

        if epoch % opt.test_interval == 0:
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(test_generator):
                with torch.no_grad():
                    images = data[0]
                    targets = data[1]
                    images = list(image for image in images)
                    targets = [{k: v for k, v in t.items()} for t in targets]

                    if torch.cuda.is_available():
                        losses = model(images.cuda(), targets.cuda())
                        cls_loss, reg_loss = losses['loss_classifier'], losses['loss_box_reg']
                    else:
                        losses = model(images, targets)
                        cls_loss, reg_loss = losses['loss_classifier'], losses['loss_box_reg']

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
                torch.save(model.state_dict(), os.path.join(opt.saved_path, "fasterrcnn_resnet50_waymo.pth"))
#                 torch.save(model, os.path.join(opt.saved_path, "fasterrcnn_resnet50_waymo.pth"))

                dummy_input = torch.rand(opt.batch_size, 3, 512, 512)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                if isinstance(model, nn.DataParallel):
                    model.module.backbone_net.model.set_swish(memory_efficient=False)

                    torch.onnx.export(model.module, dummy_input,
                                      os.path.join(opt.saved_path, "fasterrcnn_resnet50_waymo.onnx"),
                                      verbose=False)
                    model.module.backbone_net.model.set_swish(memory_efficient=True)
                else:
                    model.backbone_net.model.set_swish(memory_efficient=False)

                    torch.onnx.export(model, dummy_input,
                                      os.path.join(opt.saved_path, "fasterrcnn_resnet50_waymo.onnx"),
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
