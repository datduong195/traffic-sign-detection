# python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to classify objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import picamera
import can
from PIL import Image
from tflite_runtime.interpreter import Interpreter

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():
  global prev_label_id
  tempDict = {"1":0,"2":0,"3":0,"4":0}
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  
  print(height,width)
  frame = 0
  with picamera.PiCamera(resolution=(640, 320), framerate=30) as camera:    ### (640,480) -> (640,320)
    camera.start_preview()
    try:
      stream = io.BytesIO()
      count = 0
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize((width, height),
                                                         Image.ANTIALIAS)
        #if frame % 2 != 0:
        #    continue
        start_time = time.time()
        results = classify_image(interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000
        label_id, prob = results[0]
        if prob < 0.5:
            label_id = 4
        tempDict[str(label_id)] +=1
        if(count == 10):
          tempMax = max(tempDict.items())
          for index in tempDict.keys():
            if (tempDict[index] == tempMax):
              canSend(int(index))
          count = 0
          tempDict = {"1":0,"2":0,"3":0,"4":0}
        ################### LABEL_ID   0,1,2,3,4 --> 40limit, 40-nolimit, cross, stop, zbackground    
        
        print(labels[label_id], label_id, prob,elapsed_ms)
        stream.seek(0)
        stream.truncate()
        camera.annotate_text = '%s %d %.2f\n%.1fms' % (labels[label_id], label_id, prob,
                                                    elapsed_ms)
        count+=1
    finally:
      camera.stop_preview()

def canSend(label_id):
  messageCAN = [0,0,0,0,0,0,0,0]
  #if(label_id != prev_label_id):
  messageCAN[2] = label_id
  bus = can.interface.Bus(bustype='socketcan',
                        channel='can0',
                        bitrate=500000)
  #CAN ID 0x2A = 42
  message = can.Message(arbitration_id=42, data=messageCAN)
  bus.send(message)
  prev_label_id = label_id
  time.sleep(1)
if __name__ == '__main__':
  main()
