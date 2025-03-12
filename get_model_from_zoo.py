import urllib.request as rlib


base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/person-vehicle-bike-detection-2000/FP32/"

model_xml = "person-vehicle-bike-detection-2000.xml"
model_bin = "person-vehicle-bike-detection-2000.bin"

rlib.urlretrieve(base_url + model_xml , "Models//" + model_xml)
rlib.urlretrieve(base_url + model_bin , "Models//" + model_bin)


print("models has been downloaded")
