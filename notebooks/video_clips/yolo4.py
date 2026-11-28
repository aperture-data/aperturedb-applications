import sys
import cv2
import argparse
import random
import time
import os
import os.path as osp

import urllib3
from pathlib import Path
import hashlib
from tqdm import tqdm

# source: https://github.com/kingardor/YOLOv4-OpenCV-CUDA-DNN
class YOLOv4:

    def __init__(self,args=None):
        """ Method called when object of this class is created. """

        self.args = None
        self.net = None
        self.names = None

        self.args = self.parse_arguments() if args is None else args
        self.initialize_network()
        self.run_inference()

    def parse_arguments(self):
        """ Method to parse arguments using argparser. """

        parser = argparse.ArgumentParser(description='Object Detection using YOLOv4 and OpenCV4')
        parser.add_argument('--image', type=str, default='', help='Path to use images')
        parser.add_argument('--stream', type=str, default='', help='Path to use video stream')
        parser.add_argument('--cfg', type=str, default='models/yolov4.cfg', help='Path to cfg to use')
        parser.add_argument('--weights', type=str, default='models/yolov4.weights', help='Path to weights to use')
        parser.add_argument('--namesfile', type=str, default='models/coco.names', help='Path to names to use')
        parser.add_argument('--input_size', type=int, default=416, help='Input size')
        parser.add_argument('--use_gpu', default=False, action='store_true', help='To use NVIDIA GPU or not')
        parser.add_argument('--outdir', type=str, default="video", help='Location to put the output')
        parser.add_argument('--no-squash-detections', action='store_true', help="Fail if detections ouput exists") 

        return parser.parse_args()

    def initialize_network(self):
        """ Method to initialize and load the model. """

        self.net = cv2.dnn_DetectionModel(self.args.cfg, self.args.weights)
        
        if self.args.use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        if not self.args.input_size % 32 == 0:
            print('[Error] Invalid input size! Make sure it is a multiple of 32. Exiting..')
            sys.exit(0)
        self.net.setInputSize(self.args.input_size, self.args.input_size)
        self.net.setInputScale(1.0 / 255)
        self.net.setInputSwapRB(True)
        with open(self.args.namesfile, 'rt') as f:
            self.names = f.read().rstrip('\n').split('\n')

        if not osp.exists( self.args.outdir ):
            os.makedirs( self.args.outdir )

    def image_inf(self):
        """ Method to run inference on image. """

        frame = cv2.imread(self.args.image)

        timer = time.time()
        classes, confidences, boxes = self.net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
        print('[Info] Time Taken: {}'.format(time.time() - timer), end='\r')
        
        if(not len(classes) == 0):
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%s: %.2f' % (self.names[classId], confidence)
                left, top, width, height = box
                b = random.randint(0, 255)
                g = random.randint(0, 255)
                r = random.randint(0, 255)
                cv2.rectangle(frame, box, color=(b, g, r), thickness=2)
                cv2.rectangle(frame, (left, top), (left + len(label) * 20, top - 30), (b, g, r), cv2.FILLED)
                cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255 - b, 255 - g, 255 - r), 1, cv2.LINE_AA)

        cv2.imwrite('result.jpg', frame)
        cv2.imshow('Inference', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return

    def stream_inf(self):
        """ Method to run inference on a stream. """

        source = cv2.VideoCapture(0 if self.args.stream == 'webcam' else self.args.stream)

        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)

        i = 0
        total_start = time.time()
        detection_file = Path(self.args.outdir).joinpath('detections.csv')
        if self.args.no_squash_detections and detection_file.exists():
            print(f"Detections exists ({detection_file}), and arguments request no overwriting")
            return
        csv_file = open ( detection_file,"a") 
        pbar = tqdm( unit=' Frames', desc=f"Detecting" ) 
        while(source.isOpened()):
            ret, frame = source.read()
            if not ret and self.args.stream != 'webcam':
                break
            if ret:
                timer = time.time()
                classes, confidences, boxes = self.net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
                pbar.set_description(desc="Detecting: Last Time = {:.2}s".format(time.time() - timer),refresh=False)
                pbar.update()
                

                if(not len(classes) == 0):
                    for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                        className = self.names[classId]
                        left, top, width, height = box
                        csv_file.write(f"{i},{className},{confidence},{left},{top},{width},{height}\n")

                cv2.imwrite(osp.join( self.args.outdir,'video%d.jpg'%i),frame)
                i = i + 1
        total_end = time.time()
        csv_file.close()
        print("Done! %d frame%s, %f seconds" % ( i, "s" if i != 1 else "", total_end - total_start))

    def run_inference(self):

        if self.args.image == '' and self.args.stream == '':
            print('[Error] Please provide a valid path for --image or --stream.')
            sys.exit(0)

        if not self.args.image == '':
            self.image_inf()

        elif not self.args.stream == '':
            self.stream_inf()

        #cv2.destroyAllWindows()



if __name__== '__main__':

    yolo = YOLOv4.__new__(YOLOv4)
    yolo.__init__()


class RemoteYOLOv4(YOLOv4):
    root="https://aperturedata-public.s3.us-west-2.amazonaws.com/aperturedb_applications/"
    files={
            "coco.names":"634a1132eb33f8091d60f2c346ababe8b905ae08387037aed883953b7329af84",
            "yolov4-tiny.cfg": "6cbf5ece15235f66112e0bedebb324f37199b31aee385b7e18f0bbfb536b258e",
            "yolov4-tiny.weights": "cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87",
            "yolov4.cfg": "a15524ec710005add4eb672140cf15cbfe46dea0561f1aea90cb1140b466073e",
            "yolov4.weights": "8463fde6ee7130a947a73104ce73c6fa88618a9d9ecd4a65d0b38f07e17ec4e4"
            }
    chunk_size = 1024*128
    output_path="./models"

    def __init__(self,args=None):
        print("RemoteYOLO v4")
        self.http = urllib3.PoolManager()
        for f in self.files.keys():
            disk_path=Path(self.output_path).joinpath(f)
            disk_path.absolute().parent.mkdir( exist_ok=True)
            self.download_file( self.root+f, disk_path, self.files[f])
        print("All YOLOv4 model files downloaded")
        super().__init__(args)

    def download_file(self, from_path, to_path, expected_sha256 ):
        if to_path.exists():
            with open( to_path,"rb") as hash_file:
                sha = hashlib.sha256()
                while True:
                    data = hash_file.read(1024*64)
                    if not data:
                        break
                    sha.update(data)
                if sha.hexdigest() == expected_sha256:
                    return
                else:
                    print(f"Digest = {sha.hexdigest()} vs {expected_sha256}: Hash failed, aborting!") # we could delete and redownload.
                    sys.exit(1)

        req = self.http.request('GET', from_path, preload_content=False)
        with open( to_path, 'wb') as out:
            pbar = tqdm( unit='B',unit_scale=True, desc=f"{to_path.name}", total=int( req.headers['Content-Length'] ))
            while True:
                data = req.read(self.chunk_size)
                if not data:
                    break
                pbar.update(len(data))
                out.write(data)
        req.release_conn()
