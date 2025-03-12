# Urban_object_detection_NCS2_Raspberry
Urban_object_detection_NCS2_Raspberry


# the official implementation of Efficient Urban Object Detection on Edge Devices: A Study on Raspberry Pi 4 and Intel NCS2



# Intstall OpenVino on Raspberry Pi 4
in this project you first need to install OpenVino on Rasbperry pi. To address this should know that the latest version which supports NCS2 is 2022.3.2 LTS. after running bellow instruction you should 
remember that whenever you want to use openvino you should run steps 5-7. Furthermore, openvino on this version supports only Python 3.7 and 3.9


1) go to https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3.2/linux/
2) download l_openvino_toolkit_debian9_2022.3.2.9279.e2c7e4d7b4d_armhf.tgz if your os is 32bit otherwise download l_openvino_toolkit_debian9_2022.3.2.9279.e2c7e4d7b4d_arm64.tgz
3) unzip it in your <PATH>
4) go to the <PATH>
5) open terminal in your current <PATH> and run "source setupvar.sh"

6) run "cd install_dependencies"
7) run "./install_NCS_udev_rules.sh"

# after installing Openvino you
1) first run the get_model_from_zoo.py to get model you need to download from openVino zoo libraries
2) to test performance and fps in both CPU and Myriad you can run TEST_CPU_GPU.py





