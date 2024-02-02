# Neural Network Pruning Algorithm Based on L0-norm Sparse Optimization
As Convolutional Neural Networks (CNNs) continue to grow in size, the demand for computing resources and storage capacity increases exponentially. 
This growth significantly hampers the deployment of models in resource-constrained environments. 
Channel pruning, a potent technique for diminishing the complexity and computational requirements of CNNs, offers a solution to this challenge. 
This paper introduces two channel pruning methods based on L0-norm sparse optimization, named L0-norm Pruner and Automatic L0-norm Pruner. 
The L0-norm Pruner formulates channel pruning as a sparse optimization problem involving the L0-norm. 
Inspired by the problem-solving process, a Zero Norm (ZN) module is proposed. 
This module can autonomously select the output channels for each layer within the model according to a preset global channel pruning ratio. 
This method enables precise control over the pruning ratio, achieving effective pruning of complex models.
Furthermore, to further enhance the model's performance after pruning, we also develop the Automatic L0-norm Pruner. 
This method employs the artificial bee colony algorithm to adjust channel pruning ratios, thus mitigating the negative impact of manually preset channel pruning ratios on model performance.
Our experiments demonstrate that the proposed pruning methods outperform several state-of-the-art approaches.

## Code Structure
```
├───Auto_L0_pruner  
│       ├───main_auto_cifar.py                  (Search and fine-tune for CIFAR-10 dataset using Auto_L0_pruner)   
│       ├───main_auto_imagenet.py               (Search and fine-tune for ImageNet dataset using Auto_L0_pruner)  
├───L0_pruner  
│       ├───main_even_cifar.py                  (Train for CIFAR-10 dataset using L0_pruner)
│       ├───main_even_cifar_ft.py               (Fine-tune for CIFAR-10 dataset using L0_pruner)
│       ├───main_even_imagenet.py               (Train for ImageNet dataset using L0_pruner)
│       ├───main_even_imagenet_ft.py            (Fine-tune for ImageNet dataset using L0_pruner)
├───models                                      (CNN models)
├───utils  
│       ├───get_params_flops.py                 (Get parameters Flops of the original model and pruned model)
│       ├───resprune_resnet50.py                (Prune resnet50 after training for ImageNet dataset using L0_pruner)
│       ├───resprune_resnet56_resnet_110.py     (Prune resnet56/110 after training for CIFAR-10 dataset using L0_pruner)
│       ├───vggprune.py                         (Prune vgg after training for CIFAR-10 dataset using L0_pruner)
```
## Dataset
### CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
### ImageNet Dataset: https://image-net.org/challenges/LSVRC/2012
