# SMCL
This is a PyTorch implementation of Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles.

# Results
We provide the results for image classification using CIFAR10 dataset. We have used ResNet-50 network for our experiments.
Here ensemble size M = 5 and the number of predictors K = 1.
The code can be easily extended for any dataset and experiments as well as for semantic segmentation and image captioning tasks.

| Ensemble Size M | Oracle Accuracy |
| --------------- | --------------- |
|     1           |      85.42      |
|     2           |       89.71     |
|     3           |       92.40     |
|     4           |       93.11     |
|     5           |       94.79     |
