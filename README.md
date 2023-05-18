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
| 6 | October, 11 | <b>Seminar 3:</b> Latent Variable Models. Gaussian Mixture Model (GMM). GMM and MLE. ELBO and EM-algorithm. GMM via EM-algorithm. | [notebook](seminars/seminar3/seminar3.ipynb) | [video](https://youtu.be/vU4oZIMCEs4) |
| 7 | October, 18 | <b>Lecture 4:</b> VAE limitations. Posterior collapse and decoder weakening. Tighter ELBO (IWAE). Normalizing flows prerequisities.  | [slides](lectures/lecture4/Lecture4.pdf) | [video](https://youtu.be/pHpHnERLB_Y) |
| 8 | October, 25 | <b>Seminar 4:</b> VAE implementation hints. IWAE theory. | [notebook](seminars/seminar4/seminar4.ipynb) | [video](https://youtu.be/q3mzc8Vm_34) |
| 9 | November, 1 | <b>Lecture 5:</b> Normalizing Flow (NF) intuition and definition. Forward and reverse KL divergence for NF. Linear flows. | [slides](lectures/lecture5/Lecture5.pdf) | [video](https://youtu.be/l2o0T_A8Zvc) |
| 10 | November, 8 | <b>Seminar 5:</b> Flows. Planar flows. Forward KL vs Reverse KL. Planar flows via Forward KL and Reverse KL. | [notebook](seminars/seminar5/seminar5.ipynb)<br>[planar_flow_practice](seminars/seminar5/planar_flow.ipynb)<br>[autograd_jacobian](seminars/seminar5/jacobian_note.ipynb) | [video](https://youtu.be/Sa6SvShVrwM) |
| 11 | November, 15 | <b>Lecture 6:</b> Autoregressive flows (gausian AR NF/inverse gaussian AR NF). Coupling layer (RealNVP). NF as VAE model. | [slides](lectures/lecture6/Lecture6.pdf) | [video](https://youtu.be/5zQgNGd2Ss8) |
| 12 | November, 22 | <b>Seminar 6:</b> RealNVP implementation hints. Integer Discrete Flows | [notebook_part1](seminars/seminar6/seminar6_part1.ipynb)<br>[notebook_part2](seminars/seminar6/seminar6_part2.ipynb) | [video](https://youtu.be/12athcbZYSU) |
| 13 | November, 29 | <b>Lecture 7:</b> Discrete data vs continuous model. Model discretization (PixelCNN++). Data dequantization: uniform and variational (Flow++). ELBO surgery and optimal VAE prior. Flow-based VAE prior. | [slides](lectures/lecture7/Lecture7.pdf) | [video](https://youtu.be/b0f-w6F0NOU) |
| 14 | December, 6 | <b>Seminar 7:</b>  Discretization of continuous distribution (MADE++). Aggregated posterior distribution in VAE. VAE with learnable prior. | [notebook_part1](seminars/seminar7/seminar7_part1.ipynb)<br>[notebook_part2](seminars/seminar7/seminar7_part2.ipynb) | [video](https://youtu.be/Y1cMl9aG84A) |
| 15 | February, 7 | <b>Lecture 8:</b> Flows-based VAE posterior vs flow-based VAE prior. Likelihood-free learning. GAN optimality theorem. | [slides](lectures/lecture8/Lecture8.pdf) | [video](https://youtu.be/ISO3udXBf-I) |
| 16 | February, 14 | <b>Seminar 8:</b> Glow implementation. Vanilla GAN in 1D coding. | [VanillaGAN_todo](seminars/seminar8/GAN_colab.ipynb)<br>[VanillaGAN_done](seminars/seminar8/GAN_colab_with_code.ipynb)<br>[Glow](seminars/seminar8/Glow.ipynb) | [video](https://youtu.be/M8h1b2QOD44) |
| 17 | February, 21 | <b>Lecture 9:</b> Vanishing gradients and mode collapse, KL vs JS divergences. Adversarial Variational Bayes. Wasserstein distance. Wasserstein GAN (WGAN). | [slides](lectures/lecture9/Lecture9.pdf) | [video](https://youtu.be/mbHiRvvTF3Q) |
| 18 | February, 28 | <b>Seminar 9:</b> KL vs JS divergences. Mode collapse. Vanilla GAN on multimodal 1D and 2D data. Wasserstein distance theory. | [notebook](seminars/seminar9/seminar9.ipynb)<br>[WGAN_theory](seminars/seminar9/Continuous_1_wasserstein_note.pdf) | [video](https://youtu.be/rTtl7UqCos4) |
| 19 | March, 7 | <b>Lecture 10:</b> WGAN with gradient penalty (WGAN-GP). Spectral Normalization GAN (SNGAN). f-divergence minimization. GAN evaluation. | [slides](lectures/lecture10/Lecture10.pdf) | [video](https://youtu.be/00pPubghlA4) |
| 20 | March, 14 | <b>Seminar 10:</b> WGANs on multimodal 2D data. GANs zoo. Evolution of GANs. StyleGAN implementation. | [notebook_todo](seminars/seminar10/seminar10_colab.ipynb)<br>[notebook_done](seminars/seminar10/seminar10.ipynb)<br>[GANs_evolution](seminars/seminar10/GANs_evolution_and_StyleGAN.pdf)<br>[StyleGAN](seminars/seminar10/StyleGAN) | [video](https://youtu.be/-qT0N0r2TyI) |
| 21 | March, 21 | <b>Lecture 11:</b> GAN evaluation (Inception score, FID, Precision-Recall, truncation trick). Discrete VAE latent representations. | [slides](lectures/lecture11/Lecture11.pdf) | [video](https://youtu.be/je_8yIm0a1M) |
| 22 | March, 28 | <b>Seminar 11:</b> StyleGAN coding and assessing. Unpaired I2I translation. CycleGAN: discussion and coding. | [notebook_todo](seminars/seminar11/seminar11_colab.ipynb)<br>[notebook_done](seminars/seminar11/seminar11_solutions.ipynb) | [video](https://youtu.be/PHyTmxRAtcY) |
| 23 | April, 4 | <b>Lecture 12:</b> Vector quantization, straight-through gradient estimation (VQ-VAE). Gumbel-softmax trick (DALL-E). Neural ODE.  | [slides](lectures/lecture12/Lecture12.pdf) | [video](https://youtu.be/vw7r5oPcIqY) |
| 24 | April, 11 | <b>Seminar 12:</b> Beyond GANs: Neural Optimal Transport: theory and practice. VQ-VAE implementation hints. | [notebook](seminars/seminar12/seminar12.ipynb)<br>[NOT_theory](seminars/seminar12/NOT_note.pdf);<br>[NOT](https://github.com/iamalexkorotin/NeuralOptimalTransport) seminar<br> by [Alex Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en):<br>[notebook](https://github.com/iamalexkorotin/NeuralOptimalTransport/blob/main/seminars/NOT_seminar_strong.ipynb), [solutions](https://github.com/iamalexkorotin/NeuralOptimalTransport/blob/main/seminars/NOT_seminar_strong_solutions.ipynb) | [video](https://youtu.be/mfkk6Og7m4A) |
| 25 | April, 18 | <b>Lecture 13:</b> Adjoint method. Continuous-in-time NF (FFJORD, Hutchinson's trace estimator). Kolmogorov-Fokker-Planck equation and Langevin dynamic. SDE basics. | [slides](lectures/lecture13/Lecture13.pdf) | [video](https://youtu.be/ylovOLxhrj8) |
| 26 | April, 25 | <b>Seminar 13:</b> CNF theory. Langevin Dynamics. Energy-based Models. | [notebook](seminars/seminar13/seminar13.ipynb) | [video](https://youtu.be/yaIw5F9wsV8) |
| 27 | May, 2 | <b>Lecture 14:</b> Score matching. Noise conditioned score network (NCSN). Gaussian diffusion process. | [slides](lectures/lecture14/Lecture14.pdf) | [video](https://youtu.be/OcKwsXDRvNA) |
| 29 | May, 16 | <b>Lecture 15:</b> Denoising diffusion probabilistic model (DDPM): objective, link to VAE and score matching. | [slides](lectures/lecture15/Lecture15.pdf) | [video](https://youtu.be/BTA2YAwyV2A) |
| 28 | May, 16 | <b>Seminar 14:</b> NCSN and DDPM : theory and implementation on 2D data. | TBA | TBA |
|  | June, ??? | <b>Oral exam</b> |  |  |

## Homeworks
| Homework | Date | Deadline | Description | Link |
|---------|------|-------------|--------|-------|
| 1 | September, 28 | October, 12 | <ol><li>Theory (KDE, MADE, alpha-divergences).</li><li>PixelCNN on MNIST.</li><li>PixelCNN autocomplete and receptive field.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw1.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-MIPT-course/blob/main/homeworks/hw1.ipynb) |
| 2 | October, 12 | October, 26 | ImageGPT. | Available at the course chat |
| 3 | October, 26 | November, 9 | <ol><li>Theory (log-derivative trick, IWAE theorem, EM-algorithm for GMM).</li><li>VAE on 2D data.</li><li>VAE on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw3.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-MIPT-course/blob/main/homeworks/hw3.ipynb) |
| 4 | November, 9 | November, 23 | ResNetVAE on CIFAR10. | Available at the course chat |
| 5 | November, 23 | December, 7 | <ol><li>Theory (Sylvester flows).</li><li>RealNVP on 2D data.</li><li>RealNVP on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw5.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-MIPT-course/blob/main/homeworks/hw5.ipynb) |
| 6 | December, 7 | December, 16 | <ol><li>Theory (MI in ELBO surgery).</li><li>VAE with AR decoder on MNIST.</li><li>VAE with AR prior on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw6.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-MIPT-course/blob/main/homeworks/hw6.ipynb) |
| 7 | February, 22 | March, 9 | Vanilla GAN on CIFAR10. | Available at the course chat |
| 8 | March, 9 | March, 23 | <ol><li>Theory (IW dequantization, LSGAN, GP theorem).</li><li>WGAN/WGAN-GP/SN-GAN on CIFAR10.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw8.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-MIPT-course/blob/main/homeworks/hw8.ipynb) |
| 9 | March, 23 | April, 6 | f-GAN on CIFAR10. | Available at the course chat |
| 10 | April, 6 | April, 20 | <ol><li>Theory (Neural ODE Pontryagin theorem, Gumbel-Max trick).</li><li>FID and Inception Score.</li><li>VQ-VAE with PixelCNN prior.</li></ol> | [![Open In Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=Open%20in%20Github&color=lightgrey)](homeworks/hw10.ipynb)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/r-isachenko/2022-2023-DGM-MIPT-course/blob/main/homeworks/hw10.ipynb) |
| 11 | April, 20 | May, 10 | Continuous-in-time Normalizing Flows in 2D. | Available at the course chat |
| 12 | May, 10 | May, 24 | TBA | TBA |

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
