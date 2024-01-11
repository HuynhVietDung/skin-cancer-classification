# skin_cancer_classification

This is my Final Project in Univerisity: Melanoma Skin Cancer Classification

About the data
Melanoma Skin Cancer Dataset contains 10000 images. Melanoma skin cancer is deadly cancer, early detection and cure can save many lives. This dataset will be useful for developing the deep learning models for accurate classification of melanoma. Download Dataset at https://www.dropbox.com/scl/fi/aco2jcp6kkzij0v9f61dc/data.zip?rlkey=iak9wd4t96t3opo85lqeq7bfz&dl=0.

## Part 1: Data Preparation
* Download Data and Setting the Seed for Reproducibility
* Defined and invoked set_seed function to ensure consistency across runs.
* Configured random seeds for numpy, random, and torch.
### Data Directory Preparation
* Created a validation directory at /content/melanoma_cancer_dataset/validation/.
* Utilized an existing training directory at /content/melanoma_cancer_dataset/train/.
* Formed separate sub-directories for each label in the validation set.
### Data Splitting (Train-Validation)
* Split the dataset with an 8:2 ratio between training and validation. Moveing 20% of images from each label in the training set to the * validation set.
* Employed train_test_split for a randomized and representative split.
### DataModule Class Implementation
* Developed DataModule class, a subclass of LightningDataModule.
* Constructor accepts batch size, number of workers, file paths, and optional transformations.
### Data Transformations
* Defined transformations for resizing, normalizing, and tensor conversion.
* Additional training data augmentations include random rotation, flips, and color jitter.
### Data Loader Configuration
* setup method in DataModule prepares datasets with respective transformations.
* Data loaders for training, validation, and testing datasets are instantiated.
* Instantiated training and validation data loaders for efficient data handling.
## Part 2: CNN Architectures
We built 3 models( AlexNet, VGG16 and ResNet50 ) and applied Data Augmentation to find the best model. We used Accuracy as the key metric to evaluate these models. The results are summarized in Part 3.

![](https://github.com/HuynhVietDung/skin_cancer_classification/blob/main/Method.png)
### AlexNet Architecture
* Layer Configurations:
- First Layer: Input 3, Output 96, Kernel 11x11, Stride 4, Padding 2, ReLU Activation, MaxPooling.
- Second Layer: Input 96, Output 256, Kernel 5x5, Padding 2, ReLU Activation, MaxPooling.
- Third to Fifth Layers: Varying channels (256 to 384), Kernel 3x3, Padding 1, ReLU Activation, MaxPooling on Fifth Layer.
* Common Components:
- Optimizer: Adam (Learning Rate: 1e-3).
- Loss Function: Cross-entropy.
### VGG16 Architecture
* Layer Details:
- Convolutional Layers: Based on VGG16 configuration.
-  Adaptive Average Pooling: Output size 7x7.
- Fully Connected Layers: First Layer 512x7x7, Second Layer 4096, Third Layer matches number of classes.
* Common Components:
- Optimizer: Adam (Learning Rate: 1e-3).
- Loss Function: Cross-entropy.
### ResNet Architecture
* Configuration:
- Backbone Network: Pre-trained torchvision.models.resnet50.
- Feature Extractor: Sequential layers from ResNet-50, excluding the final fully connected layer.
- Classifier: Linear layer for two-class output.
* Common Components:
- Optimizer: Adam (Learning Rate: 1e-3).
- Loss Function: Cross-entropy.
## Part 3: Model Results
* Number of batch size: 32
* Number of epoch: 10

* Without Data Augmentation
  
| Model	| AlexNet |	VGG16 |	ResNet  |
| ------ | ------ | ------ | ------ |
| Val Acc	| 91.09 |	90.43 |	91.7 |
| Test Acc |	90.8	| 90.57 |	92.28 |

* With Data Augmentation
  
| Model |	AlexNet |	VGG16 |	ResNet |
| ----- | ------ | ----- | ------ |
| Val Acc |	86.75 |	87.18 |	91.04 |
| Test Acc |	87.91 |	85.03 |	91.27 |

## Part 4: Conclusion
There is no big difference between the results of the models. Applying data augmentation methods also does not help increase the accuracy of the models. The highest result is ResNet50 without Data Augmentation get highest Accuracy 91.7% on Trainset and 92.28% Accuracy, 95.87% Recall, 89.89% Precision and 92.61% F1Score on Testset . This is also the main model that we want to propose for this problem of Melanoma Classification.
