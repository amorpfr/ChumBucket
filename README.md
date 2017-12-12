# ChumBucket
Project Applied Machine Learning 2017 (Classifying Plankton species) <\br
#### Note that the data of the train images are not included in this zip file for storage purposes.

# Files
The code for classyfing plankton species consists the following files:

#### pre.py
Script that will preprocess the images suitbale for feature extracting. 

#### surf.py
Script that will extract SURF features for each image.

#### features.py
Script that will extract gloabal features for each image.

#### training.py
Script that wll train a model based on the train images and features.

#### test.py
Script that is able to create a submission for Kaggle.

#### visualization.py
Script to visualise results and others.


# How to Run
The files are structured as a pipeline.

#### For the training set
1. Run pre.py (set test=False) 	-> 	input: image paths, output: preprocess.pkl 
2. Run surf.py 			-> 	input: preprocess.pkl, output: surf.pkl 
3. Run features.py 		-> 	input: surf.pkl, output: features.pkl 
4. Run training.py 		->	input: features.pkl, output: model.pkl
 

#### For the test set

5. Run pre.py (set test=True) 	-> 	input: image paths, output: preprocess_test.pkl 
6. Run surf.py 			-> 	input: preprocess_test.pkl, output: surf_test.pkl 
7. Run features.py 		-> 	input: surf_test.pkl, output: features_test.pkl 

#### Create submission for Kaggle
8. Run test.py			-> 	input: features.pkl, features_test.pkl, output: subission.csv
