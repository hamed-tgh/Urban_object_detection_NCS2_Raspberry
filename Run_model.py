from openvino.runtime import Core
import numpy as np
import cv2
import time


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

image = cv2.imread("2.jpg")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

input_shape = input_layer.shape
img_resized = cv2.resize(image, (input_shape[3],input_shape[2]))
image_input = np.expand_dims(img_resized.transpose(2,0,1), axis = 0)
image_input = image_input.astype(np.float32)

result = compiled_model([image_input])[output_layer]

for obj in result[0][0]:
	confidence = obj[2]
	if confidence > 0.2:
		x_min,y_min , x_max,y_max = (obj[3:] * [image.shape[1] , image.shape[0] ,image.shape[1], image.shape[0]]).astype(int)
		cv2.rectangle(image , (x_min,y_min), (x_max, y_max) , (0,255,0) ,2)
		label = f"{confidence:.2f}"


cv2.imshow("detection" , image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("detection has done" , result)
