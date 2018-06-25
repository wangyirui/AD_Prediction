# AD-Prediction

Convolutional Neural Networks for Alzheimer's Disease Prediction Using Brain MRI Image

## How to run the code

#### 1. train the model:
```
python main_alexnet.py --optimizer Adam --learning_rate 1e-4 --save AlexNet-fine-tune-fc-last-conv-lr1e-4 --batch_size 16 --epochs 200 --gpuid 0
or
python main_autoencoder.py --batch_size 32 --num_classes 2 --epochs 200 --gpuid 0
```


## Reference
[1] Langa KM. Is the risk of Alzheimer's disease and dementia declining? Alzheimers Res Ther. 2015;7:34.

[2] Hebert LE, Weuve J, Scherr PA and Evans DA. Alzheimer disease in the United States (2010-2050) estimated using the 2010 census. Neurology. 2013;80:1778-83

[3] Alzheimer Disease International: World Alzheimer report 2010: the global economic impact of dementia. 2010.

[4] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems. 2012.

[5] He, Kaiming, et al. Deep residual learning for image recognition. Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[6] Korolev, Sergey, et al. Residual and Plain Convolutional Neural Networks for 3D Brain MRI Classification. arXiv preprint arXiv:1701.06643 (2017).

[7] Chen, Hao, et al. Voxresnet: Deep voxelwise residual networks for volumetric brain segmentation. arXiv preprint arXiv:1608.05895 (2016).

[8] He, Kaiming, et al. Identity mappings in deep residual networks. European Conference on Computer Vision. Springer International Publishing, 2016.

[9] Tajbakhsh, Nima, et al. Convolutional neural networks for medical image analysis: Full training or fine-tuning? IEEE transactions on medical imaging 35.5 (2016): 1299-1312.

[10] Poultney, Christopher, Sumit Chopra, and Yann L. Cun. Efficient learning of sparse representations with an energy-based model. Advances in neural information processing systems. 2007.

[11] Payan, Adrien, and Giovanni Montana. Predicting Alzheimer's disease: a neuroimaging study with 3D convolutional neural networks. arXiv preprint arXiv:1502.02506 (2015).

[12] Langa KM. Is the risk of Alzheimer's disease and dementia declining? Alzheimers Res Ther. 2015;7:34.

[13] Hebert LE, Weuve J, Scherr PA and Evans DA. Alzheimer disease in the United States (2010-2050) estimated using the 2010 census. Neurology. 2013;80:1778-83.

[14] Alzheimer Disease International: World Alzheimer report 2010: the global economic impact of dementia. 2010.

[15] Chester V. Dolph, Manar D. Samad and Khan M. Iftekharuddin. Classification of Alzheimers disease using structural MRI.

[16] Yosinski J, Clune J, Bengio Y, and Lipson H. How transferable are features in deep neural networks? In Advances in Neural Information Processing Systems 27 (NIPS 14), NIPS Foundation, 2014.

[17] Pan, Sinno Jialin, and Qiang Yang. A survey on transfer learning. IEEE Transactions on knowledge and data engineering 22.10 (2010): 1345-1359.

[18] Hosseini-Asl, Ehsan, Georgy Gimel'farb, and Ayman El-Baz. Alzheimer's Disease Diagnostics by a Deeply Supervised Adaptable 3D Convolutional Network. arXiv preprint arXiv:1607.00556 (2016).

[19] Glorot, Xavier, and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. 2010.

[20] F. Liu, C. Shen. Learning Deep Convolutional Features for MRI Based Alzheimers Disease Classification. arXiv preprint arXiv:1404.3366, 2014.

[21] Sharif Razavian, Ali, et al. CNN features off-the-shelf: an astounding baseline for recognition. Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2014.

[22] Yosinski, Jason, et al. How transferable are features in deep neural networks? Advances in neural information processing systems. 2014.
