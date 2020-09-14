# Smile App
Smile App mimics a person's emotions, age, and gender from CAM, Video, and Picture and sends data to the MQTT broker. In the frontend, Smile is changing real-time.
##### App
<img src="./images/smile.jpg" width="400">

### Project Set Up and Installation 
  #### Requirements 
###  Hardware 
-   6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
-   VPU - Intel® Neural Compute Stick 2 (NCS2)
-   FPGA
###  Software 
-   Intel® Distribution of OpenVINO™ toolkit 2019 R3 release [docs](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)
-   Intel DevCloud
### Model installations
-   [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
-   [Age Gender Recognition](https://docs.openvinotoolkit.org/2019_R1/_age_gender_recognition_retail_0013_description_age_gender_recognition_retail_0013.html#:~:text=Fully%20convolutional%20network%20for%20simultaneous,not%20in%20the%20training%20set.)
-   [Emotion Recognition Model](https://docs.openvinotoolkit.org/2019_R1/_emotions_recognition_retail_0003_description_emotions_recognition_retail_0003.html)
#### Installation commands
Face Detection Model
```
python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "face-detection-adas-binary-0001" -o "your directory"\models 
```
Age Gender Recognition
```
python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "age-gender-recognition-retail-0013" -o "your directory"\models 
```
Emotion Recognition Model
```
python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name "face-detection-adas-binary-0001" -o "your directory"\models
```
#### Configuration for Python files 
##### Activate virtualenv
- ```virtualenv appenv```
- ```appenv\Scripts\activate```
In the directory run: ```pip install -r requirments.txt```
## Running the App
From the main directory:
### Start the Mosca Server
```
cd webservice/server/node-server
node ./server.js
```
You should see the following message, if successful:
```Mosca server started.```
### Start the Frontend
```
cd webservice/frontend
npm start
```
or
```
npm run dev
```
[open the link](http://localhost:3000)
### For Windows
###  Initialize OpenVINO Environment
``` 
cd C:\Program Files (x86)\IntelSWTools\openvino\bin\ 
```
```
setupvars.bat
```
In the directory run:
```
python src/app.py -fdm models/intel/face-detection-adas-binary-0001/INT1/face-detection-adas-binary-0001 -erm models/intel/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003 -agr models/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013  -i CAM -extension "{your openvino directory}/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll" -pt 0.4 -flags fdm erm agr 
```
## Documentation for running command
- -fdm for face detection model
- -erm for emotion recognition model
- -agr for age gender recognition model 
- -i is for input file (picture, video and cam)
- -extension for cpu extension which is needed for unsupported layers
- -d device type (CPU, GPU, VPU, FPGA)
- For more information you can run ```python src/app.py --help```


