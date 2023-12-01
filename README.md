# Machine Learning Project 2023 - Fashion MNIST

## Table of Contents
- [Group Members](#group-members)
- [Project Description](#project-description)
- [Data](#data)
- [Implementation](#implementation)
- [Project Structure](#project-structure)



## Group Members
- Anna Maria Gnat (https://github.com/AnnaMariaGnat)
- Josefine Nyeng (https://github.com/JosefineNyeng)
- Pedro Prazeres (https://github.com/Pheadar)



## Project Description
Exam project in the Machine Learning course for the BSc Program in Data Science at the IT University of Copenhagen, academic year 2023/24.

This is a group project, where we will explore different methods for determining the type of clothing from an
image of the item. The data for the project consists of 15,000 labelled images of clothing based on images from
the Zalando website (Xiao et al., 2017). This dataset is commonly known as the Fashion MNIST dataset.

A more detailed description of the project can be found in the Project_Description.pdf file in the Information folder.



## Data
Each image is a grayscale 28x28 picture of either a t-shirt/top, trousers, a pullover, a dress, or a shirt. The images are divided into a training set of 10,000 images and a test set of 5,000 images. The images and associated labels are available in NPY format as fashion train.npy and fashion test.npy.

Each line describes a piece of clothing. The first 784 columns are the pixel values of the 28x28 grayscale image, each taking an integer value between 0 and 255. The last column, number 785, is the category of clothing and takes values in {0, 1, 2, 3, 4}. The categories are as follows:

| Category | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Shirt |



## Implementation
The project is implemented in Python 3.11 and uses Jupyter Notebooks. The following libraries are used:
- numpy (https://numpy.org/)
- matplotlib (https://matplotlib.org/)
- scikit-learn (https://scikit-learn.org/)
- plotly (https://plotly.com/python/)
- nbformat (https://nbformat.readthedocs.io/)
- ipywidgets (https://ipywidgets.readthedocs.io/)

Required libraries can be installed using the following command:
```bash
pip install -r requirements.txt
```



## Project Structure
The project is structured as follows:
```
.
├── Datasets
│   ├── fashion_test.npy
│   └── fashion_train.npy
├── Information
│   └── Project_Description.pdf
├── Libraries
│   └── ajp.py
├── Notebooks
│   ├── 01-EDA
|   |   ├── 01-PCA.ipynb
|   |   └── 02-LDA.ipynb
│   ├── 02-Bayes
|   |   ├── 01-Bayes.ipynb
│   ├── 03-Logistic-Regression.ipynb
│   ├── 04-Decision-Trees.ipynb
│   └── 05-Neural-Networks.ipynb
├── Report
│   ├── Final_Report.pdf
│   └── figures
├── requirements.txt
└── README.md
```