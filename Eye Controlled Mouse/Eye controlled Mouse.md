    Gaze estimation

Gaze estimation is a task to predict where a person is looking. There are two directions for gaze estimation: 1) 2D gaze estimation and, 2) 3D gaze estimation. In 2D gaze estimation the task is to predict the coordinate (x, y) of gaze position on the screen (i.e., a computer screen, phone or TV), which allows the gaze point to control the screen for better human-computer interactions. In a 3D gaze estimation the 3D position of the eye-ball is found to estimate a 3D gaze vector (x,y,z) in the camera coordinate system.

2. Environment Setup

Install anaconda, setup your environment and open up your favorite IDE for coding. We will use jupyter lab.

Install mediapipe and OpenCV by inserting the following commands into your terminal.

pip install mediapipe
pip install opencv-python
3. Mediapipe — Google’s brand new ML framework for media processing

Perhaps most of you are familiar with OpenCV . It’s one of the most widely used open-source libraries of computer vision developed by Intel. OpenCV provides an optimized set of algorithms and tools for media processing including the execution of ML models and artificial neural networks. Mediapipe on the other hand, is google’s new framework for media processing that offers open-source, cross-platform, customizable ML solution for live and streaming media. The solutions offered by mediapipe are face detection /face mesh, hand detection, pose estimation, holistic, object detection, hair segmentation, motion tracking just to name a few. Mediapipe brings the heavy AI models to the embedded systems and smartphones utilizing various model optimization techniques.
3.1. The 468 face landmarks (facemesh)

Let’s jump on into the facemesh solution of mediapipe. Mediapipe’s facemesh is a 468 face landmarks solution that CPU. A map of the 468 face landmarks with their corresponding positions on the face
Now, let’s define our function to get the face landmarks from mediapipe. Insert the following code into your IDE.

Printing out a single landmark value, you will get a normalized coordinate of the image as below.

x: 0.25341567397117615
y: 0.71121746301651
z: -0.03244325891137123

Which basically is number of predicted [x,y,z] keypoints after regression. These are normalized values. We need to denormalize the keypoints to get the actual pixel coordinates in the image. To denormalize the keypoint value, multiply the x and y value of the landmarks with the width and height of the input image, respectively. (Interpreting z value is out of the scope of this article)

Insert the following code into your IDE to denormalize the (x,y) value of facemesh keypoints
3.2. Eye detection with mediapipe

Each landmark defines a specific point (a.k.a keypoints) on the image. Out of 468 keypoints of the facemesh, the index number (257, 374, 386, 362) and (159, 33, 145, 133) are the top-left and bottom-right corners of the right eye and left-eye coordinates respectively. Having the index number of the eye position on the image, we can find the eyes location.

Let’s define getRightEyeRect and getLeftEyeRect and visualize the the eyes rectangles on the image. 
4.3. Mediapipe tips:

In the figure “map facemesh keypoints on the face”, we represent Mediapipe’s 468 keypoints and their respective location on the face. Using these index number we can extract any desired part of the face. Let’s do some cools stuff using the face landmarks.
