# dlnlpfinalproject
Code_and_Notebooks folder contains all the required code for pre-processing, training, and evaluating our model.
We recommend the following order to execute notebooks:
1) Extract and Save Image Encodings - This will store all image encodings from EfficientNetB4 model into numpy arrays on your disk.
2) Data Preprocessing - This containts all necessary steps for preprocessing text (captions) data.
3) Model Training - Model architecture can be imported from model.py file and this notebook contains necessary training code. Your trained model will be saved in tensorflow's saved model format on disk.
4) Model Evaluation - You can load your pre-trained model in this notebook, and evaluate its performance on the test set. It contains necessary evaluation code and functions (for calculating BLEU scores).
