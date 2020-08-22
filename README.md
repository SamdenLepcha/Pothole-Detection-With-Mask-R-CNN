# Pothole-Detection-With-Mask-R-CNN
[![TensorFlow 1.13](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)

This repository contains the project from the article "Pothole Detection with Mask RCNN". You can find the article on my <a href="https://www.samdenlepcha.com/blogs/pothole-detection-mask-rcnn/">personal website</a> or <a href="https://medium.com/@sam.lepcha98/pothole-detection-with-mask-rcnn-b78d18a89915">medium</a>. You can find the detailed tutorial to this project in those blog articles. <br> <br>
<b> Note: The tensorflow version used for this project is 1.13.1.</b>

## Installation

<ol>
<li>Clone the entire repository into your local machine.</li>
<li>Download contents of <a href = "https://github.com/tensorflow/models/tree/r1.13.0">models</a> folder from Github and place all the contents in the folder. This is the tensorflow 1.13.1 api version</li>
<li>Place all the contents inside models from this repository inside models/research/object_detection folder.</li>
<li>Download the training configuration file from the Tensorflow Model Zoo. We are going to be using "mask_rcnn_inception_v2_coco" because of it's speed compared to the others. <a href="http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz">Download</a> it and place the extracted file also inside models/research/object_detection folder</li>
<li>Open Anaconda Command Prompt and Setup a new environment</li>
   
  ```
   C:\> conda create -n pothole pip python=3.6
  ```
<li>Activate the environment and upgrade pip </li>
  
  ```
  C:\> activate pothole
  (pothole) C:\>python -m pip install --upgrade pip
  ```

<li>All other requirements can be installed using requirements.txt</li>
  
  ```
   (pothole) C:\>pip install -r requirements.txt
  ```
 
<li>Replace "YOURPATH" below and Set The Python Path Location to where you have place the tensorflow models folder. </li>
  
  ```
  (pothole) C:\>set PYTHONPATH=YOURPATH\models;YOURPATH\models\research;D:\Projects\Pothole\MaskRCNN\models\research\slim
  ```

<li>Install the coco api library</li>
  
  ```
   (pothole) C:\>  pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
  ```

<li> After all the package installations has been done navigate to the directory where the project has been downloaded and run "app.py":</li>
  
  ```
  (pothole) C:\> python app.py
  ```
</ol>

## Results
<ol>
   <li> After running the above command you should get a screen that looks like this.</li>
   <img src="https://user-images.githubusercontent.com/33536225/90520818-764fd780-e187-11ea-91c8-2e48ece8fce2.JPG" height="60" width="600">
   <br>
   <li>Copy the url right after Running on and paste it in your browser. After selecting an image you should get this output shared below.</li> 
   <img src="https://user-images.githubusercontent.com/33536225/90521571-50770280-e188-11ea-8b83-8296cf33e6ad.png">
   <li>Final Result Ouput:</li>
   <img src="https://user-images.githubusercontent.com/33536225/90600215-c83e3f00-e213-11ea-9c7b-2382f0944d53.JPG">
</ol>

