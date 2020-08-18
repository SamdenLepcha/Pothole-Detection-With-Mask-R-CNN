# Pothole-Detection-With-Mask-R-CNN
This repository contains the project from the article "Pothole Detection with Mask RCNN". You can find the article on my <a href="https://www.samdenlepcha.com/blogs/pothole-detection-mask-rcnn/">personal website</a> or <a href="#">medium</a>. You can find the detailed tutorial to this project in those blog articles. <br> <br>
<b> Note: The tensorflow version used for this project is 1.13.1.</b>

## Installation

<ol>
<li>Clone the entire repository into your local machine.</li>
<li>Download contents of <a href = "https://github.com/tensorflow/models/tree/r1.13.0">models</a> folder from Github and place all the contents in the folder. This is the tensorflow 1.13.1 api version</li>
<li>Place all the contents inside models from this repository inside models/research/object_detection folder.</li>


  <p> Open Anaconda Command Prompt and Setup a new environment</p>
   
  ```
   C:\> conda create -n pothole pip python=3.6
  ```

  <p>Activate the environment and upgrade pip </p>
  
  ```
  C:\> activate pothole
  (pothole) C:\>python -m pip install --upgrade pip
  ```
  <p>All other requirements can be installed using requirements.txt</p>
  
  ```
   (pothole) C:\>pip install -r requirements.txt
  ```

<li> After all the package installations has been done navigate to the directory where the project has been downloaded and run "app.py":
  
  ```
  (FatigueDetection) C:\> python app.py
  ```
  <p align="center"> After running the above command you should get a screen that looks like this.

  <img src="" height="200" >
Copy the url right after Running on and paste it in your browser. 
</p>
