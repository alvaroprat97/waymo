import abc
from abc import ABC
from predict_batch import mainDataset
import torch
from torchvision import transforms
import os
root_dir = os.getcwd()

#abstract class
class IPredictor(ABC):
    #constructor
    def __init__(self, weightPath):
        self.pathPesos = weightPath

#     @abc.abstractmethod
#     def predict(self,imgPath):
#         pass

#heritage
class WaymoEfficientDetPred(IPredictor):
    def __init__(self, weightPath, version = "vanilla", scope = 'validation', break_at = 10):
        IPredictor.__init__(self, weightPath)
        self.model_type = f"EfficientDet_{version}"
        self.scope = scope
        # WP = /home/project_x/models/efficientdet/trained_models/efficientdet_waymo.pth
        self.version = version
        # Temporarily change dir
        os.chdir('/home/project_x/models/efficientdet/')
        import src.dataset as ed
        self.dataset = ed.WaymoDataset(cameras=["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_RIGHT", "SIDE_LEFT"],
                                       scope=self.scope,
                                       transform=transforms.Compose([ed.Normalizer(), ed.Resizer()]))
        if torch.cuda.is_available():
            self.model = torch.load(weightPath)
        else:
            self.model = torch.load(
                weightPath,
                map_location=torch.device('cpu'))
        # Come back to root dir
        os.chdir(root_dir)
        self.break_at = break_at
        
    def predict_full_image(self, imgpath):
        img_idx = self.dataset.filepaths.index(imgpath)
        sample = self.dataset[img_idx]
        with torch.no_grad():
            scores, annot, boxes = self.model(sample['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            boxes /= sample['scale']
#         shape = sample['img'].shape
        return boxes, annot, scores
    
    def predict(self, dataset_path, cam_type, conf):
        mainDataset(dataset_path, cam_type, conf, self, self.break_at)
        
#heritage
class WaymoRCNNPred(IPredictor):
    def __init__(self,weightPath, version = "vanilla"):
        IPredictor.__init__(self, weightPath)
        self.model_type = f"RCNN_{version}"
        self.version = version
        
    def predict(self, dataset_path, cam_type, conf):
        mainDataset(dataset_path, cam_type, conf, self)

#heritage
class WaymoMaskRCNNPred(IPredictor):
    def __init__(self, weightPath, version = "vanilla"):
        IPredictor.__init__(self, weightPath)
        self.model_type = f"MaskRCNN_{version}"
        self.version = version

    def predict(self, dataset_path, cam_type, conf):
        mainDataset(dataset_path, cam_type, conf, self)

#heritage
class WaymoYOLOPred(IPredictor):
    def __init__(self,weightPath, version = "vanilla"):
        IPredictor.__init__(self, weightPath)
        self.model_type = f"YOLO_{version}"
        self.version = version

    def predict(self, dataset_path, cam_type, conf):
        mainDataset(dataset_path, cam_type, conf, self)

# # DEFAULT
#
# class DarknetYoloPred(IPredictor):
#
#     def __init__(self,weightPath,fichNames, fichCfg):
#         IPredictor.__init__(self, weightPath)
#         self.fichNames = fichNames
#         self.fichCfg = fichCfg
#
#     def predict(self, imgPath, output, conf):
#         import detect
#         detect.mainDataset(imgPath, output, conf, self.pathPesos, self.fichNames, self.fichCfg)
#
#
# class MXnetYoloPred(IPredictor):
#
#     def __init__(self,weightPath,classes):
#         IPredictor.__init__(self, weightPath)
#         self.classes=classes
#
#     def predict(self, imgPath, output, conf):
#         import predict_batch
#         predict_batch.mainDataset(imgPath, output, conf,'yolo3_darknet53_custom', self.pathPesos, self.classes)
#
# class MXnetSSD512Pred(IPredictor):
#
#     def __init__(self,weightPath,classes):
#         IPredictor.__init__(self, weightPath)
#         self.classes=classes
#
#     def predict(self, imgPath, output, conf):
#         import predict_batch
#         predict_batch.mainDataset(imgPath, output, conf,'ssd_512_resnet50_v1_custom',self.pathPesos, self.classes)
#
# class MXnetFasterRCNNPred(IPredictor):
#
#     def __init__(self,weightPath,classes):
#         IPredictor.__init__(self, weightPath)
#         self.classes=classes
#
#     def predict(self, imgPath, output, conf):
#         import predict_batch
#         predict_batch.mainDataset(imgPath, output, conf,'faster_rcnn_resnet50_v1b_custom', self.pathPesos, self.classes)
#
# class RetinaNetResnet50Pred(IPredictor):
#
#     def __init__(self,weightPath,classes):
#         IPredictor.__init__(self, weightPath)
#         self.classes=classes
#
#     def predict(self, imgPath, output, conf):
#         import predict_batch_retinanet
#         predict_batch_retinanet.mainDataset(imgPath, output, conf,'resnet50_v1', self.pathPesos, self.classes)
#
# class MaskRCNNPred(IPredictor):
#
#     def __init__(self,weightPath,classes):
#         IPredictor.__init__(self, weightPath)
#         self.classes=classes
#
#     def predict(self, imgPath, output, conf):
#         import predict_batch_rcnn
#         predict_batch_rcnn.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)
#
#
# class Efficient(IPredictor):
#
#     def __init__(self, weightPath, classes):
#         IPredictor.__init__(self, weightPath)
#         self.classes = classes
#
#     def predict(self, imgPath, output, conf):
#         import predict_batch_efficient
#         predict_batch_efficient.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)
#
# class FSAF(IPredictor):
#
#     def __init__(self, weightPath, classes):
#         IPredictor.__init__(self, weightPath)
#         self.classes = classes
#
#     def predict(self, imgPath, output, conf):
#         import predict_batch_FSAF
#         predict_batch_FSAF.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)
#
# class FCOS(IPredictor):
#
#     def __init__(self, weightPath, classes):
#         IPredictor.__init__(self, weightPath)
#         self.classes = classes
#
#     def predict(self, imgPath, output, conf):
#         import predict_batch_FCOS
#         predict_batch_FCOS.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)
