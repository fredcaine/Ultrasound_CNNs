# A Project Report for this Breast Ultrasound CNN

Project created in full by Fredrick Farouk over the course of a month.
**THIS REPOSITORY IS NOT FOR CLINICAL USE**

---

This was one of the few projects where learning and understanding the theory took significantly longer than the actual writeup of the code. I developed a pretty strong understanding of convolutional neural networks, common practices to train deep learning models, and PyTorch.
I also learned to create figure sets for my work, as is displayed in `Figure Set/`.

---

## Objective

The main goal was to classify ultrasound images into either:
- normal  
- containing (a) benign lesion(s)  
- containing (a) malignant lesion(s)

This was done by a CNN parametrized as ResNet-18, and followed as many ideas as possible set out in this research paper:  
https://doi.org/10.1038/s41467-021-26023-2

---

## Differences from the Paper

Some things I implemented differently to the paper were:
- finding one hyperparameter at a time, as opposed to using random guesses (I described training my hyperparameters more in detail in hyperparameter_optimisation_logs.txt),  
- forcing the model to rely on just one image, as no (open-source) dataset allowed me to implement an attention mechanism,  
- using an 4:0:1 ratio for training:evaluation:validation as opposed to the paper's 6:3:1, and  
- using a much smaller dataset.

***For more detail on the exact details of training and evaluating, as well as the structure of the model, see the flowcharts in `Figure Set/Model Flowchart.svg`***

---

## Results

In the end, the model achieved a **59.15%** accuracy and an AUROC of **0.7404** on the BUS-UCLM dataset. These results can be explained in a few ways:

Firstly, the dataset simply isn't large enough to support such a complex task.  

Secondly, the training dataset is different to the evaluation dataset, meaning any differences in the way both datasets format their images will lead to issues in the model guessing.  

Thirdly, the evaluation dataset (BUS-UCLM) features heavy markings, as well as an indicator of which breast is being examined.  

---

This is likely the biggest issue:  
(https://doi.org/10.1186/s12859-023-05444-4) tells us that CNNs are very sensitive to markings, and since these are scattered randomly across the image, no amount of preprocessing can really help.

---

## Additional Evaluation

My only available option that wasn't marked is the training dataset itself. Although some contamination may be present, it is not substantial as training *augments* the image aggressively while evaluating does not.

**Results in the Figure Set were based off this evaluation.**

This led to an accuracy of **92.05%**, with an AUROC of **0.972**

***For a more detailed analysis of the model's saliency mapping, see `Figure Set/Saliency Map Examples.png`***
***For more detail on the AUROC values, see `Figure Set/ROC Curve.svg`***

---

## Usage

To run, clone this repository and run `evaluation.py`.

---

If you would like to play around with some other models from `alternate_saliency_models/`, rename them to `"saliency_model.pth"` and place them in the root directory, replacing `saliency_model.pth`. I have chosen the most optimised one though.

---

If you would like to try training a model on your own hyperparameters, change them in `saliency_training.py` then run the file. You will find a `checkpoint.pth` created. You can evaluate your checkpoint by replacing `saliency_model.pth` with it. (Of course, you can store `saliency_model.pth`, my own optimised result, elsewhere.)
