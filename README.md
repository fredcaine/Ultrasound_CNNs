Project created in full by Fredrick Farouk.

Created over the course of a month.

This project was one of the few projects where learning and understanding the theory took significantly longer than the actual writeup of the code. I developed a pretty strong understanding of convolutional neural networks, common practices to train deep learning models, and usage of PyTorch.

The main goal was to classify ultrasound images into either normal, containing benign lesions or containing malignant lesions.
This was done by a CNN parametrized as ResNet-18, and followed every idea set out in this research paper: https://doi.org/10.1038/s41467-021-26023-2

Due to the lack of a(n open-source) dataset that follows the required properties, I did not add the attention mechanism. I also trained only 10 models when optimising hyperparameters (as shown in hyperparemeter_optimization_logs.txt) as opposed to 30.

To run, clone this repository and run evaluation.py.

If you would like to play around with some other models from alternate_saliency_models/, rename them to "saliency_model.pth" and place them in the root directory, replacing saliency_model.pth. I have chosen the most optimised one though.
