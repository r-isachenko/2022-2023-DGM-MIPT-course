# Deep Generative Models course, MIPT, 2022-2023

## Description
The course is devoted to modern generative models (mostly in the application to computer vision). 

We will study the following types of generative models: 
- autoregressive models, 
- latent variable models, 
- normalization flow models, 
- adversarial models,
- diffusion models.

Special attention is paid to the properties of various classes of generative models, their interrelationships, theoretical prerequisites and methods of quality assessment.

The aim of the course is to introduce the student to widely used advanced methods of deep learning.

The course is accompanied by practical tasks that allow you to understand the principles of the considered models.

## Contact the author to join the course or for any other questions :)

- **telegram:** [@roman_isachenko](https://t.me/roman_isachenko)
- **e-mail:** roman.isachenko@phystech.edu

## Materials

| # | Date | Description | Slides | Video |
|---|---|---|---|---|
| 1 | September, 6 | <b>Lecture 1:</b> Logistics. Generative models overview and motivation. Problem statement. Divergence minimization framework. Autoregressive modelling. | [slides](lectures/lecture1/Lecture1.pdf) | [video](https://youtu.be/6iJnWstvn3g) |
| 2 | September, 13 | <b>Seminar 1:</b> Introduction. Maximum likelihood estimation. Histograms. Kernel density estimation (KDE). | [notebook](seminars/seminar1/seminar1.ipynb) | [video](https://youtu.be/6RGjcdNwT-8) |
| 3 | September, 20 | <b>Lecture 2:</b> Autoregressive models (WaveNet, PixelCNN). Bayesian Framework. Latent Variable Models (LVM). Variational lower bound (ELBO). | [slides](lectures/lecture2/Lecture2.pdf) | [video](https://youtu.be/BNVvMZvs_VM) |
| 4 | September, 27 | <b>Seminar 2:</b> MADE theory and practice. PixelCNN implementation hints. Gaussian MADE. | [notebook](seminars/seminar2/seminar2.ipynb) | [video](https://youtu.be/etX4zcThxgM) |
| 5 | October, 4 | <b>Lecture 3:</b> EM-algorithm, amortized inference. ELBO gradients, reparametrization trick. Variational Autoencoder (VAE). | [slides](lectures/lecture3/Lecture3.pdf) | [video](https://youtu.be/544zO_mYcg4) |
| 6 | October, 11 | <b>Seminar 3:</b> Latent Variable Models. Gaussian Mixture Model (GMM). GMM and MLE. ELBO and EM-algorithm. GMM via EM-algorithm | [notebook](seminars/seminar3/seminar3.ipynb) | [video](https://youtu.be/vU4oZIMCEs4) |
| 7 | October, 18 | <b>Lecture 4:</b> VAE limitations. Posterior collapse and decoder weakening. Tighter ELBO (IWAE). Normalizing flows prerequisities.  | [slides](lectures/lecture4/Lecture4.pdf) | TBA |
| 8 |  | <b>Seminar 4:</b> EM-algorithm. VAE theory. Automatic differentiation through random graph. | TBA | TBA |
| 9 |  | <b>Lecture 5:</b> Flow models definition. Forward and reverse KL divergence. Linear flows (Glow). Residual flows (Planar/Sylvester flows). | TBA | TBA |
| 10 |  | <b>Seminar 5:</b> IWAE theory. IWAE variational posterior. VAE vs Normalizing flows. | TBA | TBA |
| 11 |  | <b>Lecture 6:</b> Autoregressive flows (MAF/IAF). Coupling layer (RealNVP). | TBA | TBA |
| 12 |  | <b>Seminar 6:</b> Planar flows. Forward vs Reverse KL. | TBA | TBA |
| 13 |  | <b>Lecture 7:</b> Uniform and variational dequantization. ELBO surgery and optimal VAE prior. Flows-based VAE posterior vs flow-based VAE prior. | TBA | TBA |
| 14 |  | <b>Seminar 7:</b> VAE prior (VampPrior). SurVAE. RealNVP hints. | TBA | TBA |
| 15 |  | <b>Lecture 8:</b> Disentanglement learning (beta-VAE, DIP-VAE + summary). Likelihood-free learning. GAN theorem. | TBA | TBA |
| 16 |  | <b>Seminar 8:</b> GAN vs VAE vs NF. GAN in 1d coding. | TBA | TBA |
| 17 |  | <b>Lecture 9:</b> Vanishing gradients and mode collapse, KL vs JSD. Adversarial Variational Bayes. Wasserstein distance. | TBA | TBA |
| 18 |  | <b>Seminar 9:</b> GAN vs VAE theory. KL vs JS divergences. | TBA | TBA |
| 19 |  | <b>Lecture 10:</b> Wasserstein GAN. WGAN-GP. Spectral Normalization GAN. f-divergence minimization. | TBA | TBA |
| 20 |  | <b>Seminar 10:</b> WGAN: practice. Optimal transport task. SN-GAN: practice. | TBA | TBA |
| 21 |  | <b>Lecture 11:</b> GAN evaluation (Inception score, FID, Precision-Recall, truncation trick). GAN models (Self-Attention GAN, BigGAN, PGGAN, StyleGAN). | TBA | TBA |
| 22 |  | <b>Seminar 11:</b> StyleGAN: implementation hints. | TBA | TBA |
| 23 |  | <b>Lecture 12:</b> Discrete VAE latent representations. Vector quantization, straight-through gradient estimation (VQ-VAE). Gumbel-softmax trick (DALL-E). Neural ODE. | TBA | TBA |
| 24 |  | <b>Seminar 12:</b> NeuralODE explanation. | TBA | TBA |
| 25 |  | <b>Lecture 13:</b> Adjoint method. Continuous-in-time NF (FFJORD, Hutchinson's trace estimator). Kolmogorov-Fokker-Planck equation and Langevin dynamic. SDE basics. | TBA | TBA |
| 26 |  | <b>Seminar 13:</b> TBA | TBA | TBA |
| 27 |  | <b>Lecture 14:</b> Score matching. Noise conditioned score network (NCSN). Denoising diffusion probabilistic model (DDPM). | TBA | TBA |
| 28 |  | <b>Seminar 14:</b> TBA | TBA | TBA |
|  |  | <b>Oral exam</b> | TBA | TBA |

## Homeworks 
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | September, 28 | October, 12 | <ol><li>Theory (KDE, MADE, alpha-divergences).</li><li>PixelCNN on MNIST.</li><li>PixelCNN autocomplete and receptive field.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw1.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-MIPT-course/blob/main/homeworks/hw1.ipynb) |
| 2 | October, 12 | October, 26 | ImageGPT | Available at the course chat |
| 3 |  |  | <ol><li>Theory (log-derivative trick, IWAE theorem).</li><li>VAE on 2D data.</li><li>VAE on CIFAR10.</li></ol> | TBA |
| 4 |  |  | TBA | Available at the course chat |
| 5 |  |  | <ol><li>Theory (Sylvester flows).</li><li>RealNVP on 2D data.</li><li>RealNVP on CIFAR10.</li></ol> | TBA |
| 6 |  |  | TBA | Available at the course chat |
| 7 |  |  | <ol><li>Theory (MI in ELBO surgery).</li><li>VAE with AR decoder on MNIST.</li><li>VAE with AR prior on CIFAR10.</li></ol> | TBA |
| 8 |  |  | TBA | Available at the course chat |
| 9 |  |  | <ol><li>Theory (IW dequantization, LSGAN).</li><li>WGAN/WGAN-GP on 2D data.</li><li>WGAN-GP on CIFAR10.</li></ol> | TBA |
| 10 |  |  | TBA | Available at the course chat |
| 11 |  |  | <ol><li>Theory (Neural ODE backprop).</li><li>SN-GAN on CIFAR10.</li><li>FID and Inception Score.</li></ol> | TBA |
| 12 |  |  | TBA | Available at the course chat |

## Game rules
- 6 homeworks each of 13 points = **78 points**
- oral cozy exam = **26 points**
- maximum points: 78 + 26 = **104 points**
### Final grade: `floor(relu(#points/8 - 2))`

## Prerequisities
- probability theory + statistics
- machine learning + basics of deep learning
- python + basics of one of DL frameworks (pytorch/tensorflow/etc)

## Previous episodes
- [2022, autumn, AIMasters](https://github.com/r-isachenko/2022-2023-DGM-AIMasters-course)
- [2022, spring, OzonMasters](https://github.com/r-isachenko/2022-DGM-Ozon-course)
- [2021, autumn, MIPT](https://github.com/r-isachenko/2021-DGM-MIPT-course)
- [2021, spring, OzonMasters](https://github.com/r-isachenko/2021-DGM-Ozon-course)
- [2020, autumn, MIPT](https://github.com/r-isachenko/2020-DGM-MIPT-course)
