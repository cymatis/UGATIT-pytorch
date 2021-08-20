## UGATIT with 2 GPUs

> Example Plot of A2B Translation Process
![UGATIT_GPU](https://user-images.githubusercontent.com/63994269/130188522-ea77b1e8-5fdf-49f1-bdc6-37912403aff1.png)

This repo explicitly uses GPU:0 and GPU:1.
To use this repo, please follow one of the original author's implementation below.

### [Paper](https://arxiv.org/abs/1907.10830) | [Official Tensorflow code](https://github.com/taki0112/UGATIT) | [Official Pytorch code](https://github.com/znxlwm/UGATIT-pytorch) 

> **U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation**<br>
>
> **Abstract** *We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus on more important regions distinguishing between source and target domains based on the attention map obtained by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes between domains, our model can translate both images requiring holistic changes and images requiring large shape changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. Experimental results show the superiority of the proposed method compared to the existing state-of-the-art models with a fixed network architecture and hyper-parameters.*
