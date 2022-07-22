# Behavioral Cloning
[![Udacity Self-Driving Car Engineer Nanodegree](https://img.shields.io/badge/Udacity-Self%20Driving%20Car%20Engineer%20ND-deepskyblue?style=flat&logo=udacity)](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd0013)
[![TensorFlow](https://img.shields.io/badge/%20-Keras-grey?style=flat&logoColor=red&logo=keras)](https://keras.io/) \
In this project, I developed a self-driving car that is able to drive around different tracks by imitating human driving behaviors. 
This project can be broken into 2 parts: data collection, and model training. 
During data collection, a human driver will operate the vehicle on different tracks in the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) and various sensor data will be collected for training later. 
During model training, a neural network model will be developed and trained using the data collected in the previous step so that it is able to operate the vehicle like a human being. 
After the model is trained, it will be tested in the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) to see if it is able to complete different tracks without leaving the track and collision. 
An image of the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) is shown in Fig 1.0 below.
<figure>
<img src="https://github.com/xSegFaultx/Behavioral-Cloning/raw/main/images/fig1.0.JPG" alt="Udacity Self-Driving Car Simulator">
<figcaption align = "center"><b>Fig.1.0 - Udacity Self-Driving Car Simulator </b></figcaption>
</figure>

# Data Collection
There are two tracks in the simulator. 
During the data collection phase, a human driver will drive the vehicle on both tracks in the clockwise and counter-clockwise direction. 
During driving, the following data will be collected: image from the right camera, image from the center camera, image from the left camera, throttle corresponding to the current frame, brake corresponding to the current frame, and steering angle corresponding to the current frame. 
There are 3 cameras mounted on the vehicle as illustrated in Fig 2.0 below. 
<figure>
<img src="https://github.com/xSegFaultx/Behavioral-Cloning/raw/main/images/fig2.0.PNG" alt="3-camera-system">
<figcaption align = "center"><b>Fig.2.0 - 3-Camera-System </b></figcaption>
</figure>
These 3 cameras can help the vehicle to get back to the center of the track if it drift to the side of the track. 
This is because during the data collection phase, the human driver will stay at the center of the track for the most of the time. 
As a result, the model will rarely seen any "bad driving" data, thus it will not know how to react once the car drift to the side during testing. 
With the 3 camera system, the image captured by the left and the right camera could be treated as the images that the vehicle will see if it drift to the right and the left side of the road.
We can then modify the steering angles for each image (e.g. 25 degrees to the left for the right image and 25 degrees to the right for the left image) to teach the vehicle how to steer back to the middle of the road.
Fig 2.1 shows images captured by these 3 cameras.
<figure>
<img src="https://github.com/xSegFaultx/Behavioral-Cloning/raw/main/images/fig2.1.gif" alt="Image Captured by 3 Cameras">
<figcaption align = "center"><b>Fig.2.1 - Image Captured by 3 Cameras</b></figcaption>
</figure>
This 3-camera system greatly improved the performance of the model. 
There's another thing I did that greatly improves the perfomance of the model. 
The vehicle in the simulator could be controlled by a keyboard or a controller. 
Fig 2.2 shows the distribution of steering angles yields by both controlling methods.
<figure>
<img src="https://github.com/xSegFaultx/Behavioral-Cloning/raw/main/images/fig2.2.png" alt="Keyboard VS Controller">
<figcaption align = "center"><b>Fig.2.2 - Keyboard VS Controller </b></figcaption>
</figure>
As we can see, the controller distribution is way better than the keyboard distribution. 
Since keyboards only have digital outputs, most steering angles produced by the keyboard are full left, full right or empty. 
These are not good training labels since they will make the vehicle to steer very aggrassivly and can easy drive off of the track. 
On the other hand, controllers can output analog signals which mean smaller turning angle and fine control of the vehicle



## Model
The model that I designed derives from the paper ["End to End Learning for Self-Driving Cars"](https://arxiv.org/pdf/1604.07316.pdf) from Nvidia. 
The model takes an image as input, extract features from the image using convolution layers and then map these features to steering angle (the vehicle will drive at constant speed during testing so throttle and brake are not necessary). 
Initially, I used the same exact model architecture as proposed in the paper. But the result is non-ideal. 
Soon, I found out that the model could be distracted by the features around the track. 
For example, both images in Fig 3.0 below contain dirt on the right-hand side, the model usually assume the situation in the right image is the same as the situation in the left image. 
<figure>
<img src="https://github.com/xSegFaultx/Behavioral-Cloning/raw/main/images/fig3.0.png" alt="Image Crop">
<figcaption align = "center"><b>Fig.3.0 - Images that contains similar features </b></figcaption>
</figure>
As a result, the model predicted a small turning angle and let the car drives out of the track (for the car on the left-hand side). 
To solve this problem, I decided to crope the image so that the model can focus on the features of the track.
This works quite well and since now we had a smaller input, the model trains much faster. 
Another factor that hurts the performance of the network is that the size of the input image (after cropping) is larger than the size of the image used in the paper. 
This means the model used in the paper may not be able to extract all the features from the input of this project. 
Therefore, I added 2 more convolution layers to the model to help it extracts higher-level features from the input image. 
After these changes, the model performs quite well where the vehicle can drive around both track without leaving the track or collision. 
The final structure of the model is shown in Table below.

| Layer         		|                         Description	        					                         | 
|:---------------------:|:-------------------------------------------------------------------------:| 
| Lambda        		|        normalize and 0 mean the image, input shape: 160x320x3   		        |
| Convolution 5x5		| depth 24, stride (2, 2), padding valid, activation ELU, output: 48x158x24 |
| Batch Normalization	|                                 									                                 |
| Convolution 5x5		| depth 36, stride (2, 2), padding valid, activation ELU, output: 22x77x36  |
| Batch Normalization	|                                 									                                 |
| Convolution 5x5		|  depth 48, stride (2, 2), padding valid, activation ELU, output: 9x37x48  |
| Batch Normalization	|                                 									                                 |
| Convolution 3x3		|  depth 64, stride (1, 2), padding valid, activation ELU, output: 7x18x64  |
| Batch Normalization	|                                 									                                 |
| Convolution 3x3		|  depth 80, stride (1, 1), padding valid, activation ELU, output: 5x16x80  |
| Batch Normalization	|                                 									                                 |
| Convolution 3x3		|  depth 96, stride (1, 1), padding valid, activation ELU, output: 3x14x96  |
| Batch Normalization	|                                 									                                 |
| Convolution 3x3		| depth 128, stride (1, 1), padding valid, activation ELU, output: 1x12x128 |
| Batch Normalization	|                                 									                                 |
| Flatten	      	    |                    input: 1x12x128, output: 1536 				                     |
| Fully connected		|               output: 100, activation: ELU       									                |
| Dropout				|                drop rate: 0.5        								            	                |
| Fully connected		|                output: 50, activation: ELU       									                |
| Dropout				|                drop rate: 0.5        								            	                |
| Fully connected		|                output: 10, activation: ELU       									                |
| Dropout				|                drop rate: 0.5        								            	                |
| Fully connected		|                output: 1, activation: NONE       									                |





# Result
Thanks to the simple structure of the model and the small size of the input, the model converges fairly quickly. 
It is able to drive around both track without leaving the track or collision. 
The videos of the vehicle driving around both track are shown below. 

__NOTE__: The video is sped up 8x for ease of view. If you want to check out the origin video, they are at images/track1_origin.mp4 and images/track2_origin.mp4.


# WARNING
This is a project from Udacity's ["Self-Driving Car Engineer Nanodegree"](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd0013). 
I hope my code can help you with your project (if you are working on the same project as this one) but please do not copy my code and please follow [Udacity Honor Code](https://www.udacity.com/legal/community-guidelines) when you are doing your project.

