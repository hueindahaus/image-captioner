# Image Captioner  :camera: => :computer: => :speech_balloon:

This repository presents a deep learning based system for captioning images inspired by [sgrvinod's IC-System](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) and [Bottom-Up IC-System](https://arxiv.org/pdf/1707.07998.pdf). The implementation is done in pytorch. Note that good design of the code was not the top priority due to the project's time constraint.


![a-group-of-men-playing-instruments](https://user-images.githubusercontent.com/45295311/152454995-b5b0aa15-200f-4da0-834e-d935e7e3ef42.png)

![a-black-dog-is-running-on-the-sand](https://user-images.githubusercontent.com/45295311/152454931-03dc974c-bfb2-4e25-8790-2059d091816a.png)

![two-men-are-playing-volleyball-on-a-beach](https://user-images.githubusercontent.com/45295311/152454956-7287e441-8957-4819-b247-ede5e7f2a6e0.png)


## Image captioning concept
- step 1
- step 2
- step 3...
-



## Model architecture
- Model architecture are highly inspired by [SRGAN](https://arxiv.org/abs/1609.04802) with modifications brought by [ESRGAN](https://arxiv.org/abs/1809.00219).

![gan-architecture](https://user-images.githubusercontent.com/45295311/139563996-84b435e2-8580-47c5-9b40-340f4bb592e0.png)


## Inception-residual block
- In this project, inception-residual blocks are considered instead of regular residual blocks (from SRGAN) or residual in residual dense block (from ESRGAN)

![inception-residual-block (2)](https://user-images.githubusercontent.com/45295311/139563156-970bbf47-e071-4feb-87ad-c2ece40bd13c.png)


## Other details
- Relativistic discriminator is used for a more stable and contextual training phase
- Generator perceptual loss considers pixel wise loss, feature extracted loss from a pretrained VGG19-network and adversarial loss based on discriminator's performance


## Results
- Result are presented in (PSNR/SSIM) where PSNR is peak signal to noise ratio and SSIM is structural similarity index measure.

![gan-results](https://user-images.githubusercontent.com/45295311/139563229-abbe62c1-d619-4a03-be34-7c61fd70c904.png)


