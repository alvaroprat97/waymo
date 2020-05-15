import abc
from abc import ABC

#abstract class
class IPredictor(ABC):
    #constructor
    def __init__(self, weightPath):
        self.pathPesos = weightPath

    @abc.abstractmethod
    def predict(self,imgPath):
        pass
    
# WAYMO

#heritage
class WaymoEfficientDetPred(IPredictor):
    def __init__(self,weightPath):
        IPredictor.__init__(self, weightPath)
        
    def predict(self, imgPath, output, conf):
        import EfficientDet_waymo
        EfficientDet_waymo.mainDataset(imgPath, output, conf, self.pathPesos)

#heritage
class WaymoRCNNPred(IPredictor):
    def __init__(self,weightPath):
        IPredictor.__init__(self, weightPath)
        
    def predict(self, imgPath, output, conf):
        import RCNN_waymo
        RCNN_waymo.mainDataset(imgPath, output, conf, self.pathPesos)

#heritage
class WaymoMaskRCNNPred(IPredictor):
    def __init__(self, weightPath):
        IPredictor.__init__(self, weightPath)
        
    def predict(self, imgPath, output, conf):
        import MaskRCNN_waymo
        MaskRCNN_waymo.mainDataset(imgPath, output, conf, self.pathPesos)
        
#heritage
class WaymoYOLOPred(IPredictor):
    def __init__(self,weightPath):
        IPredictor.__init__(self, weightPath)
        
    def predict(self, imgPath, output, conf):
        import YOLO_waymo
        YOLO_waymo.mainDataset(imgPath, output, conf, self.pathPesos)
        
# END WAYMO

# DEFAULT

class DarknetYoloPred(IPredictor):
    
    def __init__(self,weightPath,fichNames, fichCfg):
        IPredictor.__init__(self, weightPath)
        self.fichNames = fichNames
        self.fichCfg = fichCfg

    def predict(self, imgPath, output, conf):
        import detect
        detect.mainDataset(imgPath, output, conf, self.pathPesos, self.fichNames, self.fichCfg)


class MXnetYoloPred(IPredictor):

    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch
        predict_batch.mainDataset(imgPath, output, conf,'yolo3_darknet53_custom', self.pathPesos, self.classes)

class MXnetSSD512Pred(IPredictor):

    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch
        predict_batch.mainDataset(imgPath, output, conf,'ssd_512_resnet50_v1_custom',self.pathPesos, self.classes)

class MXnetFasterRCNNPred(IPredictor):
    
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch
        predict_batch.mainDataset(imgPath, output, conf,'faster_rcnn_resnet50_v1b_custom', self.pathPesos, self.classes)

class RetinaNetResnet50Pred(IPredictor):
    
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch_retinanet
        predict_batch_retinanet.mainDataset(imgPath, output, conf,'resnet50_v1', self.pathPesos, self.classes)

class MaskRCNNPred(IPredictor):
    
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output, conf):
        import predict_batch_rcnn
        predict_batch_rcnn.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)


class Efficient(IPredictor):

    def __init__(self, weightPath, classes):
        IPredictor.__init__(self, weightPath)
        self.classes = classes

    def predict(self, imgPath, output, conf):
        import predict_batch_efficient
        predict_batch_efficient.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)

class FSAF(IPredictor):

    def __init__(self, weightPath, classes):
        IPredictor.__init__(self, weightPath)
        self.classes = classes

    def predict(self, imgPath, output, conf):
        import predict_batch_FSAF
        predict_batch_FSAF.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)

class FCOS(IPredictor):

    def __init__(self, weightPath, classes):
        IPredictor.__init__(self, weightPath)
        self.classes = classes

    def predict(self, imgPath, output, conf):
        import predict_batch_FCOS
        predict_batch_FCOS.mainDataset(imgPath, output, conf, self.pathPesos, self.classes)
