import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'Checkbox_Trained_File'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/Check_box_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('Check_Box_Label_Map', 'Check_box_label.pbtxt')

# Number of classes to detect
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)




with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image = cv2.imread("Images/Check_Box_Input.jpeg")
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
        scores = detection_graph.get_tensor_by_name("detection_scores:0")
        classes = detection_graph.get_tensor_by_name("detection_classes:0")
        num_detections = detection_graph.get_tensor_by_name("num_detections:0")
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, 
            num_detections],  feed_dict= {image_tensor: image_np_expanded}
	)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image, np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        coordinates = vis_util.return_coordinates(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.60)
        #print(coordinates)
        length=len(coordinates)
        print(len(coordinates))
        for k in range(len(coordinates)):
            for j in range (4):
                #print(coordinates[i][j])
                if(j==0):
                    ymi=coordinates[k][j]
                if(j==1):
                    yma=coordinates[k][j]
                if(j==2):
                    xmi=coordinates[k][j]
                if(j==3):
                    xma=coordinates[k][j]
            print(xmi)
            print(xma)
            print(ymi)
            print(yma)
            crop=image[ymi:yma,xmi:xma]
            cv2.imshow("crop"+str(k),crop)
            cv2.waitKey()
        cv2.imshow("object Detection",image)
    box=np.squeeze(boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
