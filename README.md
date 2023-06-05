# templePrediction
It is a streamlit application trained on the pytorch framework of ten classes of temples which is classified.
# Temple Classification using Streamlit and PyTorch

This repository contains code for classifying images of temples into ten different classes using the Streamlit framework and the PyTorch deep learning library.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Introduction
The goal of this project is to build a temple classification system that can accurately predict the category of a given temple image. The classification model is trained using a PyTorch deep learning framework, and the web application is built using Streamlit, a Python library for creating interactive web applications.

## Installation
To use this code, you need to follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

## Usage
To start the Streamlit web application, run the following command:
```
streamlit run app.py
```
Once the application is running, you can access it in your web browser by visiting `http://localhost:8501`.

## Dataset
The dataset used for training and evaluation is not included in this repository. However, you can use your own dataset by following these guidelines:
1. Organize your dataset into a directory structure where each class has its own subdirectory. For example:
   ```
   dataset/
   ├── class1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── class2/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── ...
   ├── class10/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ```

2. Update the `config.py` file with the appropriate paths to your dataset.

## Model Training
To train the temple classification model, run the following command:
```
python train.py
```
This script will train the model using the dataset specified in the `config.py` file and save the trained model weights to the `models/` directory.

## Model Evaluation
To evaluate the trained model on a test dataset, run the following command:
```
python evaluate.py
```
This script will load the trained model from the `models/` directory, evaluate its performance on the test dataset, and display the classification accuracy.

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own purposes.
