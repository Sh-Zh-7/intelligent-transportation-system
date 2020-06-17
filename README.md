# intelligent transportation system

With the AI entering the national strategic level agenda and the continuous innovation and breakthrough of AI technology, intelligent transportation system will become an inevitable development trend in the future. To realize the intuitive understanding and convenient means of transportation system, a large number of transportation information must be processed by using computer vision technology. In the application of this technology, it not only reduces the traffic congestion, realizes the smooth traffic during the transportation, but also reduces a large number of traffic accidents, and strengthens the traffic supervision and safety.

# Prerequisites

- Windows or Linux
- **CUDA 10.0:** https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
- **cuDNN >= 7.0 for CUDA 10.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
- **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported

# Installation

- In order to run the code for demos, you need to install [ffmpeg](https://www.ffmpeg.org/).
- Clone this repo, and we'll call the directory that you cloned as ${ITS_ROOT}
- Download our pretrained model:
  - **yolo v4** on MS COCO dataset([[Google](https://drive.google.com/file/d/1eHZahK3nOQSJPveFVUKIQXZflt1Tf0ig/view?usp=sharing)]\[[Baidu](https://pan.baidu.com/s/1yHq0TX3dj80WSTljup1MtA), code: 4dm4]). Put it in `Detection/keras_yolov4/model_data/` directory.
  - **MobileNet** as encoder and **SegNet** as decoder on zebra crossing images download on the Internet([[Google](https://drive.google.com/file/d/10wvSYLTB39wKp3rSmBbfHPd93aVOjcJE/view?usp=sharing)]\[[Baidu](https://pan.baidu.com/s/19S4A1GnlzONcLxXsji-lzg), code: yv6c]). Put it in `Segmentation/segnet_mobile/` directory.
  - Other models, such as **DeepSort**'s weight, due to their  small sizes(not exceed Github's regular file's size), are already put in our repo.
- Execute the following script:

```shell
conda create -n ITS
conda activate ITS
cd ${ITS_ROOT}
pip install -r requirements.txt
```

# Run

Put your video and its background in this project and execute:

```shell
python main.py --input_video <your video> --input_background <video's background> --ouput_dir <output directory>
```

or you can just type the following script to see the demo:

```
python main.py
```

# Basic jobs

## Taffic light & zebra crossing & lane & lane mark

For those jobs, we require user to input static background image of the video, so we can get the environment’s information.

We use **object detection** and **semantic segmentation** method to get the position of traffic light and zebra crossing.

As for the lane and lane mark, we choose to use **traditional cv** method, including connected domain, contour detection, flood fill and similarity calculation, .etc.

<img src="./assets/static_jobs.png">

## Car tracking & LPR & pedestrians detection

After get the environment information about the videos, we can do further jobs that require time context information of the video.

Before this, we get the position of the traffic light, and when dealing with video, we can **convert the traffic light roi into hsv color space** to get the current color.

And the pedestrians detection is based on newly come up model: **yolo v4**(2020 May). As for the car tracking, we introduce the deep sort algorithm, which is a **tracking model** based on object detecion, so we can reduce our project’s size.

<img src="./assets/dynamic_jobs.jpg">

# Combination jobs

1. **Does the vehicle cross the line?** Judge whether the center point is higher than the lane line
2. **Does the vehicle not wait for person?** Judge whether the car and person are on the crossing line at the same time.
3. **Does the vehicle run the red light?** Judge whether the car is moving forward the the traffic light's color is red.
4. **Does the vehicle drive without guidance?** Judge whether the car's moving direction is the same as its origin lane.

You can see above that our implementation is very simple. That is because the real condition is in 3D world. But we don't have the camera's intrinsic or extrinsic matrix. So we have to judge all these conditons in 2D. So the implementation won't be too complex.

# Ackonwledement

Some part of the code is borrowed from the following repo. Thanks for their wonderful works:

1. [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT)  
2. [yehengchen/Object-Detection-and-Tracking](https://github.com/yehengchen/Object-Detection-and-Tracking)
3. [Ma-Dan/keras-yolo4](https://github.com/Ma-Dan/keras-yolo4)
4. [bubbliiing/Semantic-Segmentation](https://github.com/bubbliiiing/Semantic-Segmentation)

# Citation

### YOLO v4:
```
@misc{bochkovskiy2020yolov4,
    title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
    author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
    year={2020},
    eprint={2004.10934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### SegNet:
```
@misc{badrinarayanan2015segnet,
    title={SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation},
    author={Vijay Badrinarayanan and Alex Kendall and Roberto Cipolla},
    year={2015},
    eprint={1511.00561},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### MobileNet:
```
@misc{howard2017mobilenets,
    title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
    author={Andrew G. Howard and Menglong Zhu and Bo Chen and Dmitry Kalenichenko and Weijun Wang and Tobias Weyand and Marco Andreetto and Hartwig Adam},
    year={2017},
    eprint={1704.04861},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### Deep_SORT :

```
@inproceedings{Wojke2017simple,
title={Simple Online and Realtime Tracking with a Deep Association Metric},
author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
year={2017},
pages={3645--3649},
organization={IEEE},
doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
title={Deep Cosine Metric Learning for Person Re-identification},
author={Wojke, Nicolai and Bewley, Alex},
booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
year={2018},
pages={748--756},
organization={IEEE},
doi={10.1109/WACV.2018.00087}
}
```

# TODO

- [ ] Train the object detection model **on custom dataset**(Car dataset in surveillance scenario:[NICTA](https://research.csiro.au/data61/automap-datasets-and-code/), pedestrian dataset: [Detrac dataset](https://research.csiro.au/data61/automap-datasets-and-code/)).
- [ ] Change the deep sort model to other MOT model, for it cannot fully use the time context information.
- [ ] **Prune** and do **quantization** the model to get a higher inference speed.
- [ ] Add **car type classification** part.
- [ ] Add **car speed estimation** part after given the **intrinsic matrix**(actually it can be deduced by 3 VP) and **extrinsic matrix**(Or using Monocular Depth Estimation)

# License

[MIT License](LICENSE)

Copyright (c) 2020 sh-zh-7
