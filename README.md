# MPR-GAN: A novel neural rendering framework for MLS point cloud with deep generative learning
Qingyang Xu <br>
[Project](https://xxyxxyme.github.io/MPR_GAN_project/)
## Introduction
We propose a novel neural rendering framework based on deep generative learning for massive point cloud, named MPR-GAN. In this framework, perspective projection with intrinsic parameters scaling and nonlinear normalization is firstly utilized to transform 3D point cloud into compact 2D images, a CGAN-based rendering model is then proposed to generate photorealistic scene images from the projected 2D images. In this CGAN model, the asymmetric encoder-decoder generator can implement the inpainting and true colorization by capturing context features and emphasizing informative features, and the multi-scale discriminator is built to improve the quality of generated images with global consistence and details preservation.
