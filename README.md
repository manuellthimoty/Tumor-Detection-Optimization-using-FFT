# Tumor-Detection-Optimization-using-FFT

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## 📌 About the Project

This repository serves as the official documentation and implementation for my research project on **Mathematical Optimization in Deep Learning**. 

The primary goal is to build a **Convolutional Neural Network (CNN)** architecture completely *from scratch* to detect brain tumors from MRI scans. Unlike standard deep learning projects that rely on high-level abstractions, this project deliberately avoids optimized backends to expose and optimize the underlying linear algebra operations.

### 🚫 Frameworks Avoided
To ensure purely mathematical implementation, I **did not use**:

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

### ✅ Core Technology

Instead, the entire architecture is built using fundamental matrix operations provided by:

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## 🧠 What is a Convolutional Neural Network (CNN)?

A Convolutional Neural Network (CNN) is a class of deep neural networks, most commonly applied to analyzing visual imagery. In the context of this project, the CNN is designed to identify patterns (tumors) within MRI scans.

However, the standard implementation of CNNs faces a **computational bottleneck**:
1.  **Spatial Convolution:** The core operation involves sliding a kernel filter over an image.
2.  **Complexity:** For an image of size $N \times N$ and a kernel of size $K \times K$, the complexity is **$O(N^2 K^2)$**.
3.  **The Problem:** As the image resolution or kernel size increases, the computation time grows quadratically.

## ⚡ The Solution: Fast Fourier Transform (FFT)

This project implements the **Convolution Theorem** to optimize the forward propagation of the network. By transforming the input data from the **Spatial Domain** to the **Frequency Domain**, also know as **Discrete Fourier Transform** for this context,  we can significantly reduce computational complexity.

For detail, our main transformation can be expressed by one single equation,

$$X_k = \sum_{n=0}^N x_n e^{\frac{2i\pi nk}{N}}$$

### Mathematical Transformation:
1.  **FFT:** Transform Image ($I$) and Kernel ($K$) using Fast Fourier Transform.
2.  **Element-wise Product:** Multiply them element-wise: $\mathcal{F}(I) \odot \mathcal{F}(K)$.
3.  **IFFT:** Transform the result back using Inverse FFT.

**Result:** The complexity drops to **$O(N^2 \log N)$**, offering massive speedups for large-scale operations.


## 🚀 Getting Started

## Prerequisites

1. Python
2. Numpy
3. Matplotlib (for visualization)

## Installation

1. Clone the repo
```sh
git clone https://github.com/yourusername/Tumor-Detection-Optimization-using-FFT.git
```

2. Download the required packages
```sh
pip install requirements.txt
```
