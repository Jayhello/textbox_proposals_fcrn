# Textbox Proposals Fully Convolutional Regression Network

An implementation of the Fully Convolutional Recurrent Neural Network (FCRN) Framework in Keras as described by Ankush Gupta et. al in the paper "Synthetic Data for Text Localisation in Natural Images"


Directories containing H5Py Databases are used as input for the training and validation datasets. Each H5Py database is assumed to have records in a group called "/data" where each record's data is a numpy array containing a 512x512 grayscaled input image with an attribute called 'label' containing a 16x16x7 numpy array representing the 7 output values to regress for each cell of the input image. The 7 dimension feature vector should contain the parameters in the following order:

  (x, y, w, h, sin, cos, c)
