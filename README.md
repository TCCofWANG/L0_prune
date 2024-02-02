# l0_prune
As Convolutional Neural Networks (CNNs) continue to grow in size, the demand for computing resources and storage capacity increases exponentially. 
This growth significantly hampers the deployment of models in resource-constrained environments. 
Channel pruning, a potent technique for diminishing the complexity and computational requirements of CNNs, offers a solution to this challenge. 
This paper introduces two channel pruning methods based on \bm{$\ell_{0}$}-norm sparse optimization, named \bm{$\ell_{0}$}-norm Pruner and Automatic \bm{$\ell_{0}$}-norm Pruner. 
The \bm{$\ell_{0}$}-norm Pruner formulates channel pruning as a sparse optimization problem involving the \bm{$\ell_{0}$}-norm. 
Inspired by the problem-solving process, a Zero Norm (ZN) module is proposed. 
This module can autonomously select the output channels for each layer within the model according to a preset global channel pruning ratio. 
This method enables precise control over the pruning ratio, achieving effective pruning of complex models.
Furthermore, to further enhance the model's performance after pruning, we also develop the Automatic \bm{$\ell_{0}$}-norm Pruner. 
This method employs the artificial bee colony algorithm to adjust channel pruning ratios, thus mitigating the negative impact of manually preset channel pruning ratios on model performance.
Our experiments demonstrate that the proposed pruning methods outperform several state-of-the-art approaches.

# CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
# ImageNet Dataset: https://image-net.org/challenges/LSVRC/2012
