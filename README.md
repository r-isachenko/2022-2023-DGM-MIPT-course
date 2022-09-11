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
| 1 | September, 6 | <b>Lecture:</b> Logistics. Generative Models overview/Motivation. Problem statement. Divergence minimization framework. Autoregressive modelling. | [slides](lectures/lecture1/Lecture1.pdf) | [video](https://youtu.be/6iJnWstvn3g) |
|  | September, 13 | <b>Seminar:</b> Introduction. Density estimation in 1D. MADE theory. | TBA | TBA |
| 2 | September, 20 | <b>Lecture:</b> Autoregressive models (WaveNet, PixelCNN). Bayesian Framework. Latent Variable Models. Variational lower bound (ELBO). | [slides](lectures/lecture2/Lecture2.pdf) | [video](https://youtu.be/BNVvMZvs_VM) |
|  |  | <b>Seminar:</b> MADE practice. PixelCNN implementation hints. Bayesian inference intro, conjugate distributions. | TBA | TBA |
| 3 |  | <b>Lecture:</b> Variational lower bound. EM-algorithm, amortized inference. ELBO gradients, reparametrization trick. | TBA | TBA |
|  |  | <b>Seminar:</b> Mean field approximation. | TBA | TBA |
| 4 |  | <b>Lecture:</b> Variational Autoencoder (VAE). Posterior collapse and decoder weakening. Tighter ELBO (IWAE). | TBA | TBA |
|  |  | <b>Seminar:</b> EM-algorithm. VAE theory. Automatic differentiation through random graph. | TBA | TBA |
| 5 |  | <b>Lecture:</b> Flow models definition. Forward and reverse KL divergence. Linear flows (Glow). Residual flows (Planar/Sylvester flows). | TBA | TBA |
|  |  | <b>Seminar:</b> IWAE theory. IWAE variational posterior. VAE vs Normalizing flows. | TBA | TBA |
| 6 |  | <b>Lecture:</b> Autoregressive flows (MAF/IAF). Coupling layer (RealNVP). | TBA | TBA |
|  |  | <b>Seminar:</b> Planar flows. Forward vs Reverse KL. | TBA | TBA |
| 7 |  | <b>Lecture:</b> Uniform and variational dequantization. ELBO surgery and optimal VAE prior. Flows-based VAE posterior vs flow-based VAE prior. | TBA | TBA |
|  |  | <b>Seminar:</b> VAE prior (VampPrior). SurVAE. RealNVP hints. | TBA | TBA |
| 8 |  | <b>Lecture:</b> Disentanglement learning (beta-VAE, DIP-VAE + summary). Likelihood-free learning. GAN theorem. | TBA | TBA |
|  |  | <b>Seminar:</b> GAN vs VAE vs NF. GAN in 1d coding. | TBA | TBA |
| 9 |  | <b>Lecture:</b> Vanishing gradients and mode collapse, KL vs JSD. Adversarial Variational Bayes. Wasserstein distance. | TBA | TBA |
|  |  | <b>Seminar:</b> GAN vs VAE theory. KL vs JS divergences. | TBA | TBA |
| 10 |  | <b>Lecture:</b> Wasserstein GAN. WGAN-GP. Spectral Normalization GAN. f-divergence minimization. | TBA | TBA |
|  |  | <b>Seminar:</b> WGAN: practice. Optimal transport task. SN-GAN: practice. | TBA | TBA |
| 11 |  | <b>Lecture:</b> GAN evaluation (Inception score, FID, Precision-Recall, truncation trick). GAN models (Self-Attention GAN, BigGAN, PGGAN, StyleGAN). | TBA | TBA |
|  |  | <b>Seminar:</b> StyleGAN: implementation hints. | TBA | TBA |
| 12 |  | <b>Lecture:</b> 12. Discrete VAE latent representations. Vector quantization, straight-through gradient estimation (VQ-VAE). Gumbel-softmax trick (DALL-E). Neural ODE. | TBA | TBA |
|  |  | <b>Seminar:</b> NeuralODE explanation. | TBA | TBA |
| 13 |  | <b>Lecture:</b> Adjoint method. Continuous-in-time NF (FFJORD, Hutchinson's trace estimator). Kolmogorov-Fokker-Planck equation and Langevin dynamic. SDE basics. | TBA | TBA |
|  |  | <b>Seminar:</b> TBA | TBA | TBA |
| 14 |  | <b>Lecture:</b> Score matching. Noise conditioned score network (NCSN). Denoising diffusion probabilistic model (DDPM). | TBA | TBA |
|  |  | <b>Seminar:</b> TBA | TBA | TBA |
|  |  | <b>Oral exam</b> | TBA | TBA |

## Homeworks 
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | September, ?? | September, ?? | <ol><li>Theory (MADE, Mixture of Logistics).</li><li>PixelCNN on MNIST.</li><li>PixelCNN autocomplete and receptive field.</li></ol> |  |
| 2 |  |  | <ol><li>Theory (log-derivative trick, IWAE theorem).</li><li>VAE on 2D data.</li><li>VAE on CIFAR10.</li></ol> | TBA |
| 3 |  |  | <ol><li>Theory (Sylvester flows).</li><li>RealNVP on 2D data.</li><li>RealNVP on CIFAR10.</li></ol> | TBA |
| 4 |  |  | <ol><li>Theory (MI in ELBO surgery).</li><li>VAE with AR decoder on MNIST.</li><li>VAE with AR prior on CIFAR10.</li></ol> | TBA |
| 5 |  |  | <ol><li>Theory (IW dequantization, LSGAN).</li><li>WGAN/WGAN-GP on 2D data.</li><li>WGAN-GP on CIFAR10.</li></ol> | TBA |
| 6 |  |  | <ol><li>Theory (Neural ODE backprop).</li><li>SN-GAN on CIFAR10.</li><li>FID and Inception Score.</li></ol> | TBA |

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
