# Project: Machine Learning Edge Detection Algorithm
## Summary: This method is appropriate for edge detection of particles in microscopy images. It is less accurate than the Canny edge detection algorithm but requires 48% fewer operations, thus this makes it ideal for datasets that consist of a large number of similar images.

In this work we show how you can use the Canny edge detector to create an edge detection training set. Then, we train a logistic regression algorithm based on this data set and we present the edge detection results on the original image and on two similar images.

Open the .ipynb file for a step-by-step implementation of the algorithm and read the pdf report for more details.

The code is saved in an iPython Notebook format. To review it click on the [Canny_training_set.ipynb](./Canny_training_set.ipynb) file.

If you want to run the code in your computer you will need to follow the **Install** and **Run** instructions.

### Data

The TEM particle images here are provided by the [Priestley Polymer Laboratory](http://www.princeton.edu/cbe/people/faculty/priestley/group/) in Princeton.

**Features**  
1. Pixel intensity  
2. Sobel Derivative of Pixel intensity

**Target Variable**  
3. Edge: Each pixel can either be or not be an edge of a particle

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [OpenCV](http://opencv.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.

### Run

In a terminal or command window, navigate to the top-level project directory that contains this README and run one of the following commands:

```bash
ipython notebook Canny_training_set.ipynb
```  
or
```bash
jupyter notebook Canny_training_set.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

