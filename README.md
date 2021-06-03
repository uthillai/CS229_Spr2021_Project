# CS229_Spr2021_Project by Umesh Thillaivasan
CS299 Spr 2021 Final Project: Multi-view convolutional neural networks applied to iPhone classification

### Versions
- Python version: 3.8.5
- PyTorch version: 1.8.1

### Files and descriptions

- ```train_mvcnn.py``` : new script used to train SVCNN (Stage 1) and MVCNN (Stage 2) models 
- ```test_mvcnn.py``` : used to test SVCNN (Stage 1) and MVCNN (Stage 2) models
- models (folder)
  -   ```Model.py``` : defines the ```Model``` class
  -   ```MVCNN.py``` : defines model for stage 1 and stage 2
- tools (folder)
  -   ```ImgDataset.py``` : Loads the single view and multiview datasets, labels, and augments, resizes and normalizes the data for training and testing.
  -   ```Trainer.py``` : used to train and log each epoch of training, and also calculate and update validation accuracies throughout training
  -   ```Tester.py``` : new script to easily pass in one or multiple images into SVCNN or MVCNN models after training

### Downloading ModelNet40 Dataset
- Download images and put it under ```modelnet40_images_new_12x```: [Shaded Images (1.6GB)](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz)  

### Training and Testing
- Command for training: 
    - ModelNet40: ```python train_mvcnn.py -name mvcnn -num_models 10 -num_classes 15 -num_epochs 10 -weight_decay 0.001 -num_views 12 -cnn_name vgg11 -dataset_name modelnet40 -experiment_name modelnet40_vgg11_max```
    - iPhone dataset (not included): ```python train_mvcnn.py -name mvcnn -num_models 2 -num_classes 15 -num_epochs 10 -weight_decay 0.001 -num_views 24 -cnn_name vgg11 -dataset_name iphone -experiment_name iphone_vgg11_max```
    - Tensorboard summaries: ```tensorboard --logdir .```
  
- Command for testing:
    - For testing SVCNN with 1 image at a time: ```mvcnn_pytorch_ut_edit uthillai$ python test_mvcnn.py -num_views 1 -test_single 0 -test_network svcnn -cnn_name resnet50```
    - For testing MVCNN with multiple images at a time: ```mvcnn_pytorch_ut_edit uthillai$ python test_mvcnn.py -num_views 8 -test_single 0 -test_network mvcnn -cnn_name resnet50```
    
Notes: 
- When testing, change settings in  ```ImgDataset.py```, ```MVCNN.py```,```test_mvcnn.py``` final model paths, dataset paths, and whether max or average pooling
- test_single: 0 for full dataset test and 1 for single image
- test_network: mvcnn or svcnn
- num_views: 1 or 2 or 4, 8, etc.



## References
**A Deeper Look at 3D Shape Classifiers**  
Jong-Chyi Su, Matheus Gadelha, Rui Wang, and Subhransu Maji  
*Second Workshop on 3D Reconstruction Meets Semantics, ECCV, 2018*
https://github.com/jongchyisu/mvcnn_pytorch

**Multi-view Convolutional Neural Networks for 3D Shape Recognition**  
Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller,  
*International Conference on Computer Vision, ICCV, 2015*
https://github.com/suhangpro/mvcnn
