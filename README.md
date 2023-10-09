# Dynamic Personalized Federated Learning with Adaptive Differential Privacy

> Dynamic Personalized Federated Learning with Adaptive Differential Privacy,

> Xiyuan Yang, Wenke Huang，Mang Ye 
> *NeurIPS, 2023*
> [Link](https://neurips.cc/virtual/2023/poster/71639)

## News
* [2023-10-08] Code has released.


## Abstract
Personalized federated learning with differential privacy (DP-PFL) has been considered a feasible solution to address non-IID distribution of data and privacy leakage risks. 
However, current DP-PFL methods suffer from inflexible personalization and convergence difficulties due to two main factors: 1) Firstly, we observe that the prevailing personalization methods mainly achieve this by personalizing a fixed portion of the model, which lacks flexibility. 2) Moreover, we further demonstrate that the default gradient calculation is sensitive to the widely-used clipping operations in differential privacy, resulting in difficulties in convergence. 
Considering that Fisher information values can serve as an effective measure for estimating the information content of parameters by reflecting the model sensitivity to parameters, we aim to leverage this property to address the aforementioned challenges. 
In this paper, we propose a novel adaptive method for DP-PFL to handle these challenges. Firstly, by using layer-wise Fisher information to measure the information content of local parameters, we retain local parameters with high Fisher values during the personalization process, which are considered informative, simultaneously prevent these parameters from noise perturbation. Secondly, we introduce an adaptive approach by applying differential constraint strategies to personalized parameters and shared parameters identified in the previous for better convergence.  Our method boosts performance through flexible personalization while mitigating the slow convergence caused by clipping operations. Experimental results on CIFAR-10, FEMNIST and SVHN dataset demonstrate the effectiveness of our approach in achieving better personalization performance and robustness against clipping, under differential privacy personalized federated learning.

## Citation
```
@inproceedings{nips23dynamicPFL,
title={Dynamic Personalized Federated Learning with Adaptive Differential Privacy},
author={Yang, Xiyuan and Huang, Wenke and Ye, Mang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
}
```

## example usage 
```sh
python main_base.py
python ours.py
```

## Relevant Projects
[1] Rethinking Federated Learning with Domain Shift: A Prototype View - CVPR 2023 [[Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)]

