## UGATIT with 2 GPUs

> Example Plot of A2B Translation Process
![UGATIT_GPU](https://user-images.githubusercontent.com/63994269/130188522-ea77b1e8-5fdf-49f1-bdc6-37912403aff1.png)

This repo explicitly uses GPU:0 and GPU:1.    
This repo is based on the below official implementations and papers.    
For instructions, please follow one of the original author's implementation below.

## UGATIT
### [Paper](https://arxiv.org/abs/1907.10830) | [Official Tensorflow code](https://github.com/taki0112/UGATIT) | [Official Pytorch code](https://github.com/znxlwm/UGATIT-pytorch) 

## CartoonGAN
### [Paper](https://ieeexplore.ieee.org/document/8579084)    

## Self-Attention GAN
### [Official code](https://github.com/heykeetae/Self-Attention-GAN)    

## Requirements
* python == 3.8
* pytorch 1.19.0 or higher with CUDA 11.3
* opencv-python == 4.4.0.42

## Hardwares
* GPU : NVIDIA RTX 3090 x 2

## Dataset
* [selfie2anime dataset](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view?usp=sharing)

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
```

### Train
```
> python main.py --dataset selfie2anime --content_weight 10 --light False
```
  * Original implementations and results from paper used '--light False'.

### Test
```
> python main.py --dataset selfie2anime --phase test --light False
```

## Results

