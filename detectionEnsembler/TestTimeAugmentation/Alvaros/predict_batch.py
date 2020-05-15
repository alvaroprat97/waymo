# import the necessary packages
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths
import numpy as np
import argparse
import cv2
import os
# from utils import get_model

# TODO utils(model_type):
#    Get model

# TODO MODEL(python.model):
#   init():
#   Extracts config from utils
#   Initialise with weights from weight path

#   model.predict_full_image(image path)
#       watch out with scaling
#       model.preprocess_image
#       model.resize_image
#       predict (boxes ,scores, labels)

# def mainDataset(dataset, output, confidence, predictor):

#     classes = ["vehicle", "pedestrian", "sign", "cyclist"]

#     weightPath = predictor.weightPath
#     model_type = predictor.model_type

#     # TODO get_model method for a model wrapper class of a pytorch model
#     model = utils.get_model(weightPath, model_type)

#     imagePaths = list(paths.list_images(dataset))
#     # loop over the input image paths
#     for (i, imagePath) in enumerate(imagePaths):

#         print("[INFO] predicting on image {} of {}".format(i + 1, len(imagePaths)))

#         # load the input image (in BGR order), clone it, and preprocess it
#         image = cv2.imread(imagePath)
#         # TODO make sure that image shape is in this format (height, width, depth)
#         # TODO make sure that Cv2 is in RGB format
#         (hI, wI, d) = image.shape
#         # Make sure there are three channels
#         assert d is 3
#         # DEBUG MODE
#         imshow(image)
#         return None

#         output = image.copy()

#         # TODO preprocess_image method
#         image = model.preprocess_image(image)
#         # TODO resize_image method
#         (image, scale) = model.resize_image(image)
#         image = np.expand_dims(image, axis=0)

#         # TODO predict_full_image method
#         (boxes, scores, labels) = model.predict_full_image(image)
#         # Watch out with scale, is it reduction or augmentation?
#         boxes /= scale

#         boxes_good = []
#         for (box, score, label) in zip(boxes, scores, labels):
#             if score < confidence:
#                 continue
#             boxes_good.append((box, score, classes[label]))

#         # parse the filename from the input image path, construct the
#         # path to the output image, and write the image to disk
#         filename = imagePath.split(os.path.sep)[-1]
#         file = open(imagePath[0:imagePath.rfind(".")]+".xml", "w")
#         file.write(generateXML(imagePath[0:imagePath.rfind(".")], imagePath, wI, hI, d, boxes_good))
#         file.close()

def mainDataset(dataset_path, cam_type, confidence, model, break_at = None):

    classes = ["vehicle", "pedestrian", "sign", "cyclist"]
    cam_types = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_RIGHT", "SIDE_LEFT"]
    
#     all_cams = False
#     if cam_type in cam_types:
#         pass
#     elif cam_type is "ALL":
#         all_cams = True
#     else:
#         raise ValueError("Specifiy a correct camera label...") 
    assert cam_type in cam_types
    
    weightPath = model.pathPesos
    model_type = model.model_type

    model_name = os.path.splitext(os.path.basename(weightPath))[0] + f"_{model_type}"
    
    if cam_type is "SIDE_RIGHT" or cam_type is "SIDE_LEFT":
        # NEED TO CHANGE THIS FOR THE SMALLER IMAGES
        hI, wI, d = 886, 1920, 3
    else:
        hI, wI, d = 1080, 1960, 3 

    if not os.path.exists('salida'):
        os.mkdir('salida')
    if not os.path.exists('salida/' + model_name):
        os.mkdir('salida/'+ model_name)
#     if all_cams:
#     for cam_type in cam_types:
#     if not os.path.exists('salida/' + model_name + "/" + cam_type):
#         os.mkdir('salida/' + model_name + "/" + cam_type)
    
    output_path = os.getcwd() + '/salida/' + model_name # + "/" + cam_type
    print(f"Saving outputs at {output_path} \n")
    imagePaths = list(paths.list_files(dataset_path + "/" + cam_type))
    print(f"Accessing Images from {dataset_path}/{cam_type} \n")
    
    for (i, imagePath) in enumerate(imagePaths):
    
        print("[INFO] predicting on image {} of {} in {}".format(i + 1, len(imagePaths), cam_type))

        (boxes, labels, scores) = model.predict_full_image(imagePath)

        boxes_good = []
        for (box, score, label) in zip(boxes, scores, labels):
            if score < confidence:
                continue
            boxes_good.append((box.cpu().numpy(), score.cpu().numpy(), classes[label]))
        
        # parse the filename from the input image path, construct the
        # path to the output image, and write the image to disk
        
        filename = imagePath.split(os.path.sep)[-1]
        xml_name = filename[0:filename.rfind(".")]+ "_" + cam_type + ".xml"
        with open(output_path + "/" + xml_name, "w") as f:
            xml_strand = generateXML(xml_name, output_path, wI, hI, d, boxes_good)
            f.write(xml_strand)

        if isinstance(break_at, int) and i >= break_at - 1:
            print(f"Done with dummy {i + 1} images \n")
            return None
        
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generateXMLout(filename, outputPath, w, h, d, boxes):

    top = ET.Element('annotation')
    childFolder = ET.SubElement(top, 'folder')
    childFolder.text = 'images'
    childFilename = ET.SubElement(top, 'filename')
    childFilename.text = filename[0:filename.rfind(".")]
    childPath = ET.SubElement(top, 'path')
    childPath.text = outputPath + "/" + filename
    childSource = ET.SubElement(top, 'source')
    childDatabase = ET.SubElement(childSource, 'database')
    childDatabase.text = 'Unknown'
    childSize = ET.SubElement(top, 'size')
    childWidth = ET.SubElement(childSize, 'width')
    childWidth.text = str(w)
    childHeight = ET.SubElement(childSize, 'height')
    childHeight.text = str(h)
    childDepth = ET.SubElement(childSize, 'depth')
    childDepth.text = str(d)
    childSegmented = ET.SubElement(top, 'segmented')
    childSegmented.text = str(0)

    for box in boxes:
        label, x,y,xmax,ymax, score = box
        bbox = np.array([x,y,xmax,ymax]).astype(float)
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = label
        childScore = ET.SubElement(childObject, 'confidence')
        childScore.text = str(score)
        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'
        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'
        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'
        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(max(bbox[0],1))
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(max(bbox[1],1))
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(min(bbox[2],w-1))
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(min(bbox[3],h-1))

    return prettify(top)

def generateXML(filename, outputPath, w, h, d, boxes):

    top = ET.Element('annotation')
    childFolder = ET.SubElement(top, 'folder')
    childFolder.text = 'images'
    childFilename = ET.SubElement(top, 'filename')
    childFilename.text = filename[0:filename.rfind(".")]
    childPath = ET.SubElement(top, 'path')
    childPath.text = outputPath + "/" + filename
    childSource = ET.SubElement(top, 'source')
    childDatabase = ET.SubElement(childSource, 'database')
    childDatabase.text = 'Unknown'
    childSize = ET.SubElement(top, 'size')
    childWidth = ET.SubElement(childSize, 'width')
    childWidth.text = str(w)
    childHeight = ET.SubElement(childSize, 'height')
    childHeight.text = str(h)
    childDepth = ET.SubElement(childSize, 'depth')
    childDepth.text = str(d)
    childSegmented = ET.SubElement(top, 'segmented')
    childSegmented.text = str(0)

    for (box, score, label) in boxes:
        box = box.astype("int")
        (x, y, xmax, ymax) = box
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = label
        childScore = ET.SubElement(childObject, 'confidence')
        childScore.text = str(score)
        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'
        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'
        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'
        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(max(x,1))
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(max(y,1))
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(min(xmax,w-1))
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(min(ymax,h-1))

    return prettify(top)
