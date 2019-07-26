# Stress Distribution Inference using U-Net Based Fully Convolutional Networks

A Machine Learning approach to Finite Element Methods, using [U-Net](https://github.com/zhixuhao/unet) inspired architectures.


This project has been developed as part of my master thesis for in Imperial College London with the supervision of Prof. A. Bharath and Dr. Masouros.

![concept_diagram](https://user-images.githubusercontent.com/30337324/61964160-26730100-afc5-11e9-9c13-95d752d206fb.jpeg)

#### Practical advice

*What's missing:*

Some of the trained models were too heavy to be updated on GitHub. This means that in order to run some of the 
plotting functions you will have to run the Main and train the models yourself.
While the databases to perform ShapeToShape and CoordToShape reconstruction are generated on the fly, the ones containing
the Von Misses stresses have been produced using Marc Mentat software (a FEA software), and are not given here. If you are
interested in reproducing this work using the same data, please don't hesitate to contact me.

*How to run the code*

Run the main to train the model and display some of the key parameters to judge the performance of the model.
The main is divided into commented sections that perform different tasks (as described bellow). Comment/Uncomment the 
appropriate section to perform the task needed.

In order to print sample reconstructions there are a number of functions available:
- *Plot_Sample.py* will plot a sample reconstruction using the model you have just trained, which is saved automatically
in the folder "Saved_Models" with the name: modellino.model
- *Plot_Image_Report.py* will plot a comparison of different models for a certain task
- *Plot_Image_Report_ForceToStress.py* will plot the recontructions from the three different architectures developed here.

### Abstract

Finite Element Methods (FEM) are currently amongst the most exploited techniques to approximate the solutions of physical 
problems. While they realise the validating step for automated systems that promise to revolutionise the functional design
sector such as generative designs and personalised implants, they are extremely tedious and generally require extensive 
human intervention. Recent advancements in artificial intelligence and the wider availability of computational power have
motivated a shift towards predictive approaches using machine learning tools. However, the irregular and variable structures
of the input data and the complexity of underlying physical principles make this task remarkably complex. Previous approaches
have failed to create robust models capable of dealing with inputs as fed to FEM, which are characterised by their variable 
size and contain connectivity information. To this end, we propose two approaches: a U-Net based neural network suitable for
variable size input stress reconstruction, and a DGCNN based network able to process adjacency matrices and non-constant 
input sizes as well as infer physical properties of geometrical shapes. We introduce the Multi-Poly dataset composed of 
polygons with 4 to 8 vertices, and demonstrate the enhanced performance of U-Net models in geometrical shape reconstruction
(2.3e-5 MSE and 1.8e-4 ID) and variable-morphology input handling. In the stress reconstruction task our model outperforms 
previous ones, achieving over ten times higher accuracy with respect to ID score (0.076 ID and 17.7 AMSE). A modified version
of the Multi-Poly dataset is then used to demonstrate that our DGCNN derived model can correctly classify (98.4% accuracy)
convex and concave polygons. U-Net based models proved to be a viable alternative for stress estimation on shapes with
variable input size; DGCNN networks similarly demonstrated this ability to process/ competency in processing data of variable
size while in addition embedding an adjacency matrix. In light of these results, both may prove to be promising FEM surrogate
candidates. 

## Description

This works adapts the U-Net architecture to the tasks of shape reconstruction and stress distribution inference. The general structure of the model consists of a traditional contracting network supplemented with a series of up-sampling layers. The large number of feature channels in the up-sampling part ensures that context information and physical characteristics are propagated to higher resolution layers. 

### Architecture

A total of five architectures have been implemented and explored in this study. Three of them (Minimal, Shallow and Full U-Nets) include both an encoding and a decoding part; the other two (Half U-Net for shape reconstruction and Half U-Net for stress estimation) comprise only the decoder. Below we report the details of the two best performers of each type. 

Shallow U-Net architecture:
![UnetArch](https://user-images.githubusercontent.com/30337324/61965127-6c30c900-afc7-11e9-8c12-6290aeb9995d.png)
The Shallow U-Net takes a 64x64 matrix input and returns either the reconstructed shape of the polygon, or the distribution of stresses. Between other modifications, this network is fundamentally different from the original U-Net in the application of padding and strides, such that the input size is maintained to the output.
Its contracting path (left of figure) consists of 3 convolutional blocks each with two 3x3 padded 2D convolutions followed by a rectified linear unit (ReLU) and of a 2x2 max pooling step with stride 1 for down-sampling. Similarly, the bottleneck is made of a 3x3 padded convolution with a ReLU. In the expansive path (right side in left of Figure 8.a), every block consists of an up-sampling deconvolution of the feature map with filter size 2x2 followed by two 3x3 2D convolutions. After each up-sampling operation, there is the concatenation with the cropped feature map from the corresponding convolutional layer. The final layer consists of a 1x1 convolution to reduce the number of feature maps of the output to 1. No fully connected layer is used in the network.

Half U-Net architecture:
![Half-UnetArch](https://user-images.githubusercontent.com/30337324/61965155-7a7ee500-afc7-11e9-87ee-ac0463f31a6a.png)

The other architecture, here named Half U-Net, is designed to start from a low dimensional input and hence consists of the expansive part only. However, as this network is constrained in its input size, it presents similar limitations to previous solutions and is reported here only as an alternative approach to former studies.
It consists of 6 deconvolutional blocks, each with an up-sampling deconvolution followed by two 3x3 2D convolutions. The first 5 up-sampling operations have filter size 2x2, while the last one is applied with a 1x2 filter. After the second upsampling, a convolution with strides (2,1) is applied in order to obtain the desired output size. The final layer consists of a 1x1 convolution.

### Dataset

An ad hoc dataset, which will be referred to here as Multi-Poly dataset, was generated to train the networks. It consists of 12,000 planar shapes, equally divided between polygons with 4 to 8 corners. Vertices are generated starting from a circle of radius 1. Each vertex falls at a random distance between 0.8-1 from the centre. The angles between each vertex, initially set to 360/n, are perturbated following a normal distribution with variance (90/n)^2, where n is the number of corners for a given polygon. The coordinates are then scaled based on the canvas size (or image size).
Each polygon of the Multi-Poly dataset is described by 3 objects as follows:
- The cartesian coordinates of the polygon in the form nx2;
- A 64x64 matrix with ones in correspondence to corners and zeros otherwise;
- A 64x64 matrix with ones inside the polygon and zeros otherwise. 

![coord](https://user-images.githubusercontent.com/30337324/61965604-7e5f3700-afc8-11e9-8411-77acc86e910e.png)
![Picture1](https://user-images.githubusercontent.com/30337324/61965629-8919cc00-afc8-11e9-9e66-20998b2579c2.png)
![Picture2](https://user-images.githubusercontent.com/30337324/61965640-920a9d80-afc8-11e9-90c4-04a4ad553b4a.png)

In addition, we tested our models on the dataset created by Benkirane and Laskowski, which is composed of 10,000 pentagons with their respective Von Misses stresses. This contains information relative to corner coordinates, boundary conditions and forces applied to them, and the corresponding Von Misses stresses in the form of a 64x64 matrix (more information about the dataset can be found [here](https://github.com/mlaskowski17/Exploring-Design-Spaces)).

