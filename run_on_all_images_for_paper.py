from openvino.runtime import Core
import numpy as np
import cv2
import time
import os 



CLASSES = ['car', 'person', 'bicycle']
colors = [(0,0,255) , (255,0,0), (0,255,0)]

ie = Core()



# Check if MYRIAD is available
if "MYRIAD" not in ie.available_devices:
    print("MYRIAD device not found! Running on CPU instead.")
    device_name = "CPU"
else:
    device_name = "MYRIAD"

print(f"Running on device: {device_name}")


model_path = "/home/pars/Downloads/Python_COde/Models/person-vehicle-bike-detection-2000.xml"

model = ie.read_model(model = model_path)

compiled_model = ie.compile_model(model = model , device_name = "MYRIAD")
list_of_images = os.listdir("COCO_Images//")
print(list_of_images)

for image_name in list_of_images:
	 
	image = cv2.imread("COCO_Images//" + image_name)
	input_layer = compiled_model.input(0)
	output_layer = compiled_model.output(0)

	input_shape = input_layer.shape
	img_resized = cv2.resize(image, (input_shape[3],input_shape[2]))
	image_input = np.expand_dims(img_resized.transpose(2,0,1), axis = 0)
	image_input = image_input.astype(np.float32)

	result = compiled_model([image_input])[output_layer]

	for obj in result[0][0]:
		print("shape of object is " , obj.shape)
		print(obj)
		cals_label = obj[1].astype(int)
		confidence = obj[2]
		if confidence > 0.15:
			x_min,y_min , x_max,y_max = (obj[3:] * [image.shape[1] , image.shape[0] ,image.shape[1], image.shape[0]]).astype(int)
			color = colors[cals_label]
			
			cv2.rectangle(image , (x_min,y_min), (x_max, y_max) , color ,2)
			label = f"{confidence:.2f}"
			label = f'{CLASSES[cals_label]}'
			cv2.putText(image, label, (x_max - 10, y_max + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			


	cv2.imwrite("COCO_OUTPUT//" + image_name, image)
