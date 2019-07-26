# Exploring-Design-Spaces

A Machine Learning approach to Finite Element Methods, using [U-Net](https://github.com/zhixuhao/unet) inspired architectures.


This project has been developed as master thesis for in Imperial College London with the supervision of Prof. A. Bharath and
Dr. Masouros.

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
- *Plot_Sample.py* will plot

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


