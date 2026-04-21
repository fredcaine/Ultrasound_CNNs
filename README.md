Project created in full by Fredrick Farouk.

Created over the course of a month.

This project was one of the few projects where learning and understanding the theory took significantly longer than the actual writeup of the code. I developed a pretty strong understanding of convolutional neural networks, common practices to train deep learning models, and PyTorch.

The main goal was to classify ultrasound images into either normal, containing benign lesions or containing malignant lesions.
This was done by a CNN parametrized as ResNet-18, and followed as many ideas as possible set out in this research paper: https://doi.org/10.1038/s41467-021-26023-2

Some things I implemented differently to the paper were:
finding one hyperparameter at a time, as opposed to using random guesses,
forcing the model to rely on just one image, as no (open-source) dataset allowed me to implement an attention mechanism,
using an 4:0:1 ratio for training:evaluation:validation as opposed to the paper's 6:3:1, and
using a much smaller dataset.

In the end, the model achieved a 92.05% accuracy.

To run, clone this repository and run evaluation.py.

If you would like to play around with some other models from alternate_saliency_models/, rename them to "saliency_model.pth" and place them in the root directory, replacing saliency_model.pth. I have chosen the most optimised one though.

If you would like to try training a model on your own hyperparameters, change them in saliency_training.py then run the file. You will find a checkpoint.pth created. You can evaluate your checkpoint by replacing saliency_model.pth with it. (Of course, you can store saliency_model.pth, my own optimised result, elsewhere.)
