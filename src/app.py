import cv2
import os
import numpy as np
import logging as log
import time
import socket
import math
import argparse 
import logging
import sys 
import json
import paho.mqtt.client as mqtt

# Mqtt setting
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

from input_feeder import InputFeeder 
from face_detection import Facedetectionmodel
from emotion_recognition import EmotionRecognition 
from age_gender_recognition import AgeGenderRecognition

def get_args():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-fdm", "--face_detection_model", required=True, type=str,
                        help="Path to a face detection model xml file with a trained model")
    parser.add_argument("-erm", "--emotion_recognition", required=True, type=str,
                        help="Path to a emotion recognition model xml file with a trained model") 
    parser.add_argument("-agr", "--age_gender_recognition", required=True, type=str,
                        help="Path to a face detection model xml file with a trained model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or CAM")
    parser.add_argument("-extension", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="(CPU)-targeted custom layers. and locates in opnevino app")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="CPU, GPU, FPGA or VPU (NCS2 or MYRIAD) is acceptable")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.4,
                        help="Probability threshold for detections in the frame"
                             "(0.5 by default)")
    parser.add_argument("-flags", "--visualization", required=False, nargs='+', default=[],
                        help="flags (fdm, erm, agr)"
                        "You can see with these flags different outputs")


    return parser  

def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client() 
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)  
    return client  

def main():
    # Connect to the MQTT server
    client = connect_mqtt()
    # get args
    args = get_args().parse_args()
    flags = args.visualization
    input_file_path = args.input

    if input_file_path == "CAM":
        input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file_path): 
            logging.error("check your input file, it is not valid")
            exit(1)
        input_feeder = InputFeeder("video", input_file_path)
    
    # assigning parameters for constructor function
    # Face Detection Model
    face_detection = Facedetectionmodel(model_name=args.face_detection_model, device=args.device, threshold=args.prob_threshold, 
        extensions=args.cpu_extension) 
    # face detection model load time
    face_detection_time = time.time()
    face_detection.load_model()
    face_detection_end = time.time() - face_detection_time
    logging.error("Face Detection model load time: {:.3f} ms".format(face_detection_end * 1000)) 

    # Emotion Recogtion Model
    emotion_recognition = EmotionRecognition(model_name=args.emotion_recognition, device=args.device, 
        extensions=args.cpu_extension) 
    # emotion recognition model load time
    emotion_recognition_time = time.time()
    emotion_recognition.load_model()
    emotion_recognition_end = time.time() - emotion_recognition_time
    logging.error("Emotion Recogtion model load time: {:.3f} ms".format(emotion_recognition_end * 1000))  

    # Age Gender Recogtion Model
    age_gender_recognition = AgeGenderRecognition(model_name=args.age_gender_recognition, device=args.device, 
        extensions=args.cpu_extension) 
    # emotion recognition model load time
    age_gender_recognition_time = time.time()
    age_gender_recognition.load_model()
    age_gender_recognition_end = time.time() - emotion_recognition_time
    logging.error("Age Gender Recogtion model load time: {:.3f} ms".format(age_gender_recognition_end * 1000)) 

    logging.error("All models have been successfully loaded")
    # load inputs image, video and cam 
    input_feeder.load_data()
    counter=0 
    inference_time_start = time.time()

    for flag, frame in input_feeder.next_batch():
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        counter += 1

        # Face detection
        face_coordinates, fd_outputs = face_detection.predict(frame) 
        xmin, ymin, xmax, ymax = face_coordinates[0], face_coordinates[1], face_coordinates[2], face_coordinates[3]
    
        # croped image
        crop = frame[ymin:ymax, xmin:xmax] 

        if len(crop) != 0:
            # Emotion Recognition 
            emotion, probability = emotion_recognition.predict(crop)

            # Age Gender Recognition
            age, gender = age_gender_recognition.predict(crop)  
        
            # dwaring bounding box, axes and putting message and visualization
            """ if len(flags) != 0:
                print("") """
            cv2.rectangle(frame, (xmin, ymin), 
                        (xmax, ymax), (0, 255, 255), 1)  

            age_gender_text = "Age: {}, gender: {}".format(age, gender)
            emotion_text = "Emotion: {}, probability: {}".format(emotion, probability) 
            """ logging.error(age_gender_text)
            logging.error(emotion_text) """
            cv2.putText(frame, emotion_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, age_gender_text, (5, 50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1) 

            # publish age, gender, emotion
            client.publish("person", json.dumps({
                "gender": gender,
                "age": int(age),
                "emotion": emotion 
            })) 
        """ frame = cv2.resize(frame, (940, 640))
        cv2.imshow('frame', frame)  """

    total_time = time.time() - inference_time_start
    total_inference_time=total_time
    fps=counter/total_inference_time
    logging.error("Inference time: {:.3f}".format(total_inference_time))
    logging.error("FPS: {}".format(fps))

    # total load time
    total_model_load_time = face_detection_end + emotion_recognition_end + age_gender_recognition_time
    # getting directory name
    dirname = os.path.dirname(os.path.abspath(__file__))
    # writing all results to txt
    with open(os.path.join(dirname, '../outputs/stats.txt'), 'w') as f:
            f.write("Inference Time: {:.3f} ms".format(total_inference_time) + '\n')
            f.write("FPS: {:.1f}".format(fps) + '\n')
            f.write("All model loading times" + '\n')
            f.write("Face Detection model load time: {:.3f} ms".format(face_detection_end * 1000) + '\n') 
            f.write("Total: {:.3f} ms".format(total_model_load_time * 1000) + '\n')
     

    # Release the out writer, capture, and destroy any OpenCV windows
    input_feeder.close()
    cv2.destroyAllWindows()
    ### Disconnect from MQTT 
    client.disconnect()

if __name__ == '__main__':
    main()
