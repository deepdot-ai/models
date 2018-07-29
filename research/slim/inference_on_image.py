import numpy as np
import os
import sys
import glob
import tensorflow as tf
import time
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = '/home/xiangxin/gitclients/models/research/object_detection/ssd_mobilenet_v1_focal_loss/frozen_inference_graph.pb'
#PATH_TO_CKPT = '/home/deepdot/Dataset/check_point/faster_rcnn_resnet101/frozen_inference_graph.pb'
PATH_TO_CKPT = '/home/xiangxin/exp/res50_beer_lr_1e-3/inference/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "/home/deepdot/Dataset/Budweiser/scene_label_map.pbtxt"
NUM_CLASSES = 4

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print("Successfully loaded the model.")

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print("Successfully loaded the label map.")

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    print("Image size: %d x %d" % (im_width, im_height))
    image = image.resize([224, 224], Image.ANTIALIAS)
    (im_width, im_height) = image.size
    print("Image size: %d x %d" % (im_width, im_height))
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    np_image = np.array(image.getdata()).reshape((224, 224, 3)).astype(np.uint8)
    np_image[:,:,0] = np_image[:,:,0] - _R_MEAN;
    np_image[:,:,1] = np_image[:,:,1] - _G_MEAN;
    np_image[:,:,2] = np_image[:,:,2] - _B_MEAN;
    return np_image

PATH_TO_TEST_IMAGES_DIR = '/home/deepdot/Dataset/Budweiser/debug_images'
TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR + '/*/*.jpg')

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [ 'resnet_v2_50/predictions/Softmax', 'resnet_v2_50/predictions/Reshape_1' ]:
                tensor_name = key + ':0'
                print tensor_name
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('input:0')
            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            #output_dict['num_detections'] = int(output_dict['num_detections'][0])
            #output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            #output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['resnet_v2_50/predictions/Softmax'] = output_dict['resnet_v2_50/predictions/Softmax'][0]
            output_dict['resnet_v2_50/predictions/Reshape_1'] = output_dict['resnet_v2_50/predictions/Reshape_1'][0]
    return output_dict

for image_path in TEST_IMAGE_PATHS:
    print image_path
    image = Image.open(image_path)
    start = time.time()
    image_np = load_image_into_numpy_array(image)

    print image_np.shape
    print "First pixel"
    print "0: %d" % image_np[0][0][0]
    print "1: %d" % image_np[0][0][1]
    print "2: %d" % image_np[0][0][2]
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    end = time.time()
    print("Finished inference in %f seconds." % (end - start))
    print("resnet_v2_50/predictions/Softmax")
    print output_dict['resnet_v2_50/predictions/Softmax'].shape
    for i in range(4):
        print "score %f" % (output_dict['resnet_v2_50/predictions/Softmax'][i])
    print("resnet_v2_50/predictions/Reshape_1")
    print output_dict['resnet_v2_50/predictions/Reshape_1'].shape
    for i in range(4):
        print "score %f" % (output_dict['resnet_v2_50/predictions/Reshape_1'][i])
    # Visualization of the results of a detection.
#    vis_util.visualize_boxes_and_labels_on_image_array(
        #image_np,
        #output_dict['detection_boxes'],
        #output_dict['detection_classes'],
        #output_dict['detection_scores'],
        #category_index,
        #instance_masks=output_dict.get('detection_masks'),
        #use_normalized_coordinates=True,
        #max_boxes_to_draw=200,
        #min_score_thresh=.3,
          #line_thickness=10)
    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np)
    #plt.show()
