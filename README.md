# ml_malaria_detection
Machine Learning Python code that can detect malaria & malaria infected cells (trophozoites, rings, scizonts, infected gametocytes) in wholeslide microscopy images

00d04a90-80e5-4bce-9511-1b64eabb7a47.png: Sample wholeslide microscopy image. 1 of 1328 wholeslide images in the dataset

0ac747cd-ff32-49bf-bc1a-3e9b7702ce9c.png: Sample wholeslide microscopy image. 1 of 1328 wholeslide images in the dataset

Final paper.pdf : the paper submitted for ECE 687 Pattern Recognition at Drexel University 2019 including comparitive analysis of this project and current state of the art

NeuralClass.py: Contains the Artificial Neural Net (ANN) used in the NeuralN.py file. Created using Pytorch.

NeuralN.py: Uses the ANN created in the NeuralClass.py file for detection of malaria and malaria infected cells in wholeslide microscopy images. Maximum accuracy: 92.81 F-score: 96.25

LBP.py: (Local Binary Pattern) contains the algorithm that converts a grayscale image to a local binary pattern image

SVM.py: contains both Support Vector Machine (SVM) and Random Forest Classifier used for this dataset with and without Principal Component Analysis (PCA). SVM and Random Forest Classifier respectively: Maximum accuracy: NA (didn't converge) F-score: NA (didn't converge) Maximum accuracy: 81.89 F-score: 89.32 

plotInfectedCells.py: plots RGB of dataset, PCA histograms using Matplotlib
 


