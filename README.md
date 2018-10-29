# Matrix Factorization problem
In this repository, a classical matrix factorization algorithm has been implemented using numpy and optimizing the Mean Squared Error using Stochastic Gradient Descent. As it is a linear method, an additional version using Deep Learning techniques has been implemented in order to improve the system performance, concluding that the MSE can be decreased by an 8.5% using a very simple neural network.

## Getting started
Both algorithms results over a sample dataset are shown in the jupyter notebook located in `./notebooks/example.ipynb`. 

Experiments can be reproduced by following the notebook. For running the code the following dependencies are needed
- Pandas
- Numpy
- Tensorflow
- Python 3.6

## Results

**Matrix Factorization results**

<img width="430" alt="mse_mf" src="https://user-images.githubusercontent.com/7207415/43367863-029a5686-9354-11e8-9d94-e11e32b37136.png">

**Deep Factorization results**

<img width="424" alt="mse_df" src="https://user-images.githubusercontent.com/7207415/43367864-02b61b3c-9354-11e8-85dc-bebe0dc46825.png">

## License
This repository is licensed with MIT agreement. Copyright (c) 2018 Iván Vallés Pérez

For more information, please, check the LICENSE file.
