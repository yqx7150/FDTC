# FDTC
Frequency domain generative diffusion model for temporal compressive coherent diffraction imaging  
The Code is created based on the method described in the following paper:  
Mei, Z., Xu, Y., Lin, J., Wang, Y., Wan, W., & Liu, Q. (2024). Frequency domain generative diffusion model for temporal compressive coherent diffraction imaging. Optics and Lasers in Engineering, 181, 108424.  
https://www.sciencedirect.com/science/article/pii/S0143816624004020  
## Abstract  
Temporal compressive CDI utilizes single-exposure snapshot compression sampling in the frequency domain to visualize the dynamic process of CDI. Due to the lack of effective frequency domain prior information constraint, the traditional temporal compressive CDI method has limited imaging quality. In order to achieve high-quality dynamic reconstruction at high frame rates, a score-based diffusion model in frequency domain is used to extract the prior gradient distribution of the target in the complex frequency domain. Subsequently, this information is harmoniously woven into the physics-driven iterative phase recovery reconstruction process to restore multi-frame high-quality images. Simulation and experimental outcomes have demonstrated that the proposed technique surpasses conventional methods, delivering enhanced structural details and a reduction in artifact. The highest achieved PSNR and SSIM values are 37.25 dB and 0.997, respectively. The proposed method has equally well performance on targets across datasets, demonstrating the strong generalization of the method.
![FDTC](https://github.com/yqx7150/FDTC/blob/main/images/FDTC.jpg?raw=true)  
Flowchart of FDTC. (a) Training phase to learn the gradient distribution of complex frequency domain via denoising score matching. (b) In the reconstruction phase, learned frequency domain prior constraints are applied in the frequency domain. Spatial domain constraints, DNN-based denoising and TV are performed in the spatial domain. (c) The time-domain unfolding and reconstructed frequency-domain frames serve as the data consistency term in the reconstruction phase.  
![mnist_results](https://github.com/yqx7150/FDTC/blob/main/images/mnist_results.jpg?raw=true)  
Visual comparison of images reconstructed by each algorithm on the MNIST. (a) Ground Truth (b) TC–CDI (c) FDTC (d) FDTC+TV.  
## Requirements and Dependencies
```python==3.7.11  
Pytorch==1.7.0  
tensorflow==2.4.0  
torchvision==0.8.0  
tensorboard==2.7.0  
scipy==1.7.3  
numpy==1.19.5  
ninja==1.10.2  
matplotlib==3.5.1  
jax==0.2.26
```
## Usage
1.Download the all the files via [Baidu Drive](https://pan.baidu.com/s/10ionrg_120nQO9sJ2Iz6WA) (access code FDTC) and directly put the data in FDTC_Stage1 and FDTC_Stage2.  
## Train  
CUDA_VISIBLE_DEVICES=0 python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp --mode=train --eval_folder=result（run in FDTC_Stage2）  
## Test  
FDTC_Stage1:  
python test.py（run in FDTC_Stage1）  
FDTC_Stage2:  
python PCsampling_demo.py（run in FDTC_Stage2）  
