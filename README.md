# Gender Determination with Morphometry of Eyes 

## Introduction
Morphometry of eyes is the study of the physical characteristics of eyes, such as their size, shape, and structure. It is used to measure and analyze the differences between individuals and populations, as well as to identify certain genetic diseases. It can also be used to reveal clues about a person's age, gender, and ethnicity. Morphometric studies of eyes typically focus on the cornea, iris, sclera, and eyelids. The analysis of these features can provide insight into various aspects of eye health and vision, such as refractive errors, presbyopia, glaucoma, and cataracts. See the [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4968610/).

## Problem Statement
The anthropometric analysis of the human face is an essential study for performing craniofacial plastic and reconstructive surgeries. Facial anthropometrics are affected by various factors such as age, gender, ethnicity, socioeconomic status, environment, and region.

Plastic surgeons who undertake the repair and reconstruction of facial deformities find the anatomical dimensions of the facial structures useful for their surgeries. These dimensions are a result of the Physical or Facial appearance of an individual. Along with factors like culture, personality, ethnic background, age; eye appearance and symmetry contributes majorly to the facial appearance or aesthetics.

## Project Objective
Our objective is to build a model to scan the image of an eye of a patient and predict the gender of the patient as male or female.

## About the data
The image data used for this project was obtained from [Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/gender-determination-with-morphometry-of-eyes).

The data is in the `gender_eye` directory with the following tree:
```bash
.
├── test.csv
├── train.csv
├── test
└── train
    ├── female
    └── male

The train data contains the images for training splitted in to gender whereas test data contains unlablled images.
```

## Approach
This project employed ResNet50 pre-trained model to train a deep learning convolutional neural network (CNN) to analyze the image and determine whether a patient's eyse belongs to a male or female. The CNN model was trained with 7377 images belonging to the 2 classes and validated the model with 1843 (20%) images belonging to 2 classes.

Some parameters of the model (learning rate, inner layer size and dropout rate) were tuned before arriving at a validation accuracy of 97.3% The best model obtained achieved a precision, recall and area under the curve of roughly 96%.

The best model was converted to tensorflow lite model (eye_model.tflite).

The following software stack was used:

```bash
numpy: 1.23.4
Tensorflow keras: 2.10.0
```

## Files and Scripts
- `notebook.ipynb`: a jupyter-notebook for visualisation and analysis.
- `train.py`: modular scipt for loading the images, training the model and saving into tflite.
- `DockerFile`: a docker file for building a docker image.
- `test.py`: a script for testing the image.
- `lambda.py`: part of the script used to deploy the model into docker image.
- `Pipfile and Pipfile.lock`: `pipenv` file for virtual environment in docker.

# Build Docker
To build the docker, run the command below on your terminal or command prompt:
```bash
docker build --platform linux/amd64 -t eye-model .
```
Thereafter, you can run the docker as:
```bash
docker run -it --rm -p 9696:9696 --platform linux/amd64 eye-model:latest
```

In another terminal and same project directory, type to run the test.py script.
```bash
python test.py 
```