fingerCursor (Air Calligraphy)
==================
> Final project for EECS332 Intro. to Computer Vision

**Miaoding Dai & Evan Chien**
##### _Northwestern University, Fall 2017_

![jing.gif](https://drive.google.com/uc?export=view&id=1gV6wn6KgTVq98Lpg9K3tcdvKPLfOgD_b)

- _A simple demo of writing a Chinese character 'jing', which means 'calm' in English._

## Overview

'Finger Curosr'-like applications can be used in video games, and to control smart devices without using controller. They are also super useful in presentation.

In this project, we implemented a system that used camera to locate and track a fingertip of index finger. When we arbitrarily move the fingertip in the air (within range of camera view), trajectory of fingertip motion will be recorded and drawn in the video sequences and shown on the screen. We also add gesture recognition to the system, so the system knows when to draw and when to clean the trajectory.

Furthermore, one fancy feature is, trajectory can show different handwritings (width of trajectory) according to how fast you draw. This feature will be really interesting if you want to write some Chinese Characters (i.e. Calligraphy).

## Program Flow

![programFlow](https://drive.google.com/uc?export=view&id=1KskTkbxel2I82vKvIvmPRP-2fPDLbR7b)

## Instructions for Use

Before use the code, make sure you have Python3.5+ and OpenCV3 installed in your computer. Personally, I use Python 3.5.2 and OpenCV 3.3.0.

> For those who use Linux, Linux wheels do not support video related functionality ([link](https://pypi.python.org/pypi/opencv-contrib-python)), you may need build and install it frome source. This tutorial '[Ubuntu 16.04: How to install OpenCV](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)' may be useful for you.

After steps above are all set, everything goes straightforward. Download [fingerCursor.py](./fingerCursor.py) and [gesture2.png](./gesture2.png), and put them in the same directory. If you properly run the code, you will launch all the windows showing intermediate results of image processing.

Feel free to download them and have a try. Hope you enjoy it.

#### Tips:

1/ You may need change the device number for camera to the right one to launch your own camera.

```Python
if __name__ == '__main__':
    device = 1  # device number for camera
    fingerCursor(device)
```

2/ According to lightening condition in your environment, you may need to change the `HSV skin color segmentation mask` a little bit, tuning the system into a better behavior.

```Python
## skin color segmentation mask
skin_min = np.array([0, 40, 50],np.uint8)  # HSV mask
skin_max = np.array([50, 250, 255],np.uint8) # HSV mask
```

## Demo
Here is a [Demo video](https://drive.google.com/open?id=16cLHRbFHecWvPEc2LvOeuWlO5ktPGK1J) for your interest.


