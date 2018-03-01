# AD_Prediction

Scripts are for our Computer Vision course project "Convolutional Neural Networks for Alzheimers Disease Prediction Using Brain MRI Images".

## Preprocessing
We preprocessed brain MRI scans with segmentation, registration, normalization, noise addition and skull-stripping.

## Training
We first trained a 2D-AlexNet with transfer learning on scan slices and then combined it with SVM, where the ConvNet is a feature extractor. 
We also trained a 3D sparse auto-encoder on small patches extracted from scans to generate features, which were then inputs to a simple 3D CNN for the prediction task.

## Results
The best accuracy we got on the validation set was 86%.
