from flask import Flask, render_template, url_for, request, jsonify, abort
import os
from PIL import Image
import shutil
import sys
import numpy as np
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

sys.path.append("D:/Projects/Pothole/MaskRCNN/models/research")
sys.path.append("D:/Projects/Pothole/MaskRCNN/models/research/slim")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './Image'

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

  if request.method == 'POST':

    path = './Image'

    #Image info
    img_file = request.files['photo']
    img_name = img_file.filename
    img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
    im = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
    #Changed resolutions
    res=(800,600)
    im_resized = im.resize(res, Image.ANTIALIAS)
    im_resized.save(path+"/Resized.png")
    
    PATH_TO_FROZEN_GRAPH = './models/research/object_detection/inference_graph/frozen_inference_graph.pb'
    PATH_TO_LABELS = './models/research/object_detection/training/label.pbtxt'

    #Load frozen model onto Tensorflow memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    #Loading Label map
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    #Helper Code
    def load_image_into_numpy_array(image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


    def run_inference_for_single_image(image, graph):
      with graph.as_default():
        with tf.Session() as sess:
          # Get handles to input and output tensors
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,
                                feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict
    
    
    image = Image.open(path+"/Resized.png")
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
    
    #fig = plt.figure(figsize=IMAGE_SIZE)
    #ax = fig.gca()
    #ax.grid(False)
    #plt.imshow(image_np)
    score=output_dict['detection_scores'][0]
    cv2.imwrite("static/Output.png", image_np)

    return render_template('final.html',user_image = "Output.png",final=score)	

if __name__ == '__main__':
    app.run(debug=True, port=8000)