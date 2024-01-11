# skin_cancer_classification

This is my Final Project in Univerisity

Part 1: Data Preparation
Download Data and Setting the Seed for Reproducibility
Download Dataset at https://www.dropbox.com/scl/fi/aco2jcp6kkzij0v9f61dc/data.zip?rlkey=iak9wd4t96t3opo85lqeq7bfz&dl=0.
Defined and invoked set_seed function to ensure consistency across runs.
Configured random seeds for numpy, random, and torch.
Data Directory Preparation
Created a validation directory at /content/melanoma_cancer_dataset/validation/.
Utilized an existing training directory at /content/melanoma_cancer_dataset/train/.
Formed separate sub-directories for each label in the validation set.
Data Splitting (Train-Validation)
Split the dataset with an 8:2 ratio between training and validation. Moveing 20% of images from each label in the training set to the validation set.
Employed train_test_split for a randomized and representative split.
DataModule Class Implementation
Developed DataModule class, a subclass of LightningDataModule.
Constructor accepts batch size, number of workers, file paths, and optional transformations.
Data Transformations
Defined transformations for resizing, normalizing, and tensor conversion.
Additional training data augmentations include random rotation, flips, and color jitter.
Data Loader Configuration
setup method in DataModule prepares datasets with respective transformations.
Data loaders for training, validation, and testing datasets are instantiated.
Instantiated training and validation data loaders for efficient data handling.
Part 2: CNN Architectures
We built 3 models( AlexNet, VGG16 and ResNet50 ) and applied Data Augmentation to find the best model. We used Accuracy as the key metric to evaluate these models. The results are summarized in Part 3.
