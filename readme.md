Structure of the code:

Code Related to Yolo:
1. backbone.py: This file contains the code to build Resnet Backbone for Yolo.
2. createValDataset.py: This file contains the code to Split train.csv into train and val sets.
3. dataset.py: Contains the code to pack the data into pytorch Dataset to make it suitable for dataloader.
4. loss.py: Contains the code for loss function of yolo.
5. model.py: Contains the code to build YoloV1 model.
6. train.py: Contains the code to train the model.
7. utils.py: Contains all the helper functions used in the codebase.

Instructions to execute:
* Create a python environment and install pyTorch and other required libraries.
* Obtain the Voc dataset from Kaggle.
* run Python createValDataset.py in terminal
* run python train.py to train and get metrics of model.

Code Related to Detr:
1. main.py: Heart of the system which binds all the things together.
2. model1.py: Builds the Detr model.
3. utils1.py: Contains all the helper functions for the system.
4. evaluate.py: Contains the code for evaluating the model.
5. run python main.py to to perform visualizations of results and get metrics and pass 1 and the true_boxes as cmd arguements.

Acknowlegements:
* Youtube of Alladion Person and Yannic Kilcher were very helpful in doing our project.

Link to PPT: https://drive.google.com/drive/folders/1wZVRThEGT-Pw4T9fpPMm74m9XFDJQUsc?usp=drive_link
Link to presentation: https://drive.google.com/drive/folders/1wZVRThEGT-Pw4T9fpPMm74m9XFDJQUsc?usp=drive_link