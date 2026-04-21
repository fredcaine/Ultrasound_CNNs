# Fredrick Farouk. saliency_training.py
# This is the main file that sets up preprocessing and training of the deep-learning model (DLM).
# The model generates saliency maps that highlight important locations in the image.
# The details of training are very similar to that of
# "Artificial intelligence system reduces false-positive findings in the interpretation of breast ultrasound exams."
# This article can be found at: https://doi.org/10.1038/s41467-021-26023-2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import torch.optim as optim
import os

# ------------------------ DATASET AND PREPROCESSING ------------------------

# This is the augmentation detailed in the paper.
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),               # Randomly flip horizontally.
    transforms.RandomRotation(degrees=45),                # Randomly rotate -45 -- 45 degrees.
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.7, 1.5),
        shear=25
    ),                                                    # Random transformation explained in the paper.
    transforms.Resize((256, 256)),                        # Setting to the correct size.
    transforms.ToTensor(),                                # Conversion to a tensor, size [1, 256, 256]
    transforms.Lambda(lambda x: x.repeat(3,1,1)),         # Moving into three channels, so size [3, 256, 256]
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)       # Doesn't do anything to the image but required for training.
])

# Dataset acquired through the link in the paper:
# https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
# I understand this was the external test dataset and not the training dataset.
# There are no larger, publicly available datasets that meet the criteria of the paper's objective.
class BUSIDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('L')
        # When an item is fetched, a *new* augmentation is applied. This heavily prevents overfitting.
        if self.transform:  # This check is unnecessary and specifically avoids the annoying red squiggly line.
            img = self.transform(img)
        # Fetch the label as well to return them in a pair. Useful for data loading as seen later.
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

    # Dunder method that makes finding the dataset's length slightly more convenient.
    def __len__(self):
        return len(self.img_paths)

base_dir = "Images"
image_paths = []
labels = []

# Loading all images along with their labels.
for fname in os.listdir(os.path.join(base_dir, "normal")):
    image_paths.append(os.path.join(base_dir, "normal", fname))
    # The array in the next line is a one-hot vector with the structure: [normal, benign, malignant]
    labels.append([1,0,0])

# Similar logic for the next two loops.
for fname in os.listdir(os.path.join(base_dir, "benign")):
    image_paths.append(os.path.join(base_dir, "benign", fname))
    labels.append([0,1,0])

for fname in os.listdir(os.path.join(base_dir, "malignant")):
    image_paths.append(os.path.join(base_dir, "malignant", fname))
    labels.append([0,0,1])

# Instantiate the dataset.
dataset = BUSIDataset(image_paths, labels, augmentation_transform)

# Important step that slightly differs from the paper.
# The paper splits the dataset into 60% for training, 10% for hyperparameter tuning (validation) and 30% for evaluating the model.
# My code will instead split into 80% for training and 20% for validation, as I don't really need to evaluate the model in as much detail as the paper does.
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
# The exact type of split doesn't really matter here, so I used random split to make sure absolutely nothing stays the same.
# An example of when this might make the slightest difference is if I pause training through a checkpoint (shown later).
# This allows a different training set to absolutely ensure nothing remains standard. But again, this really doesn't make a difference due to augmentation.
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# These 'DataLoader's are structures that allow loading data in batches to accelerate training.
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
# Note, the validation set does not need to be shuffled. Even if it isn't very important, we try our best to ensure the set remains the same every run.
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

# ------------------------ MODEL DEFINITIONS ------------------------

class SaliencyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ResNet18 is the convolutional neural network (CNN) used in the paper.
        # Yes, we set pretrained to be True.
        backbone = models.resnet18(pretrained=True)

        # We now remove global average pooling (avgpool) and the fully connected layer (fc)
        # The reasoning is as follows: for avgpool, we do not want to collapse the map into 1x1, as many, many details are related to spatial context.
        # For fc, it, at the moment, produces the 1 000 pretrained classes of the original ResNet18, which are completely useless to the task at hand.
        # Otherwise, this is the standard sequencing for CNNs; convolution, batch normalization, ReLU activation, max pooling, then a few layers.
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        # This 1x1 convolution is attributed in the paper to Zhou et al., whose credit is at the link below.
        # https://doi.org/10.1109/cvpr.2016.319
        # This is run on the feature maps, the general idea being to combine low-level edges and curves into a more high-level structure.
        # The exact way this happens is much better explained in the paper linked.
        self.saliency_conv = nn.Conv2d(512, 3, kernel_size=1)
        # Just a standard sigmoid function, 1/(1+e^-x).
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # This forward pass just calls the previously defined functions in their logical order.
        h = self.features(x)
        A = self.sigmoid(self.saliency_conv(h))
        return h, A

# We choose the cost function for the DLM to be cross entropy (CE) loss, which works great for multiple channels.
criterion = nn.CrossEntropyLoss()

# This is weak supervision.
# I did consider trying a different approach of directly using the masks provided in the dataset, but I instead stayed faithful to the paper.
# top_t is the first of three optimised hyperparameters. To see how I got to these values, find "hyperparameter_optimization_logs.txt" in the root directory.
def aggregate_saliency(A, top_t=0.25):
    # We unpack the predicted y values tensor.
    batch, C, H, W = A.shape

    # Our aggregate scores will dismiss the height and width, so it's logical that the aggregate tensor is only the batch by the number of channels.
    agg = torch.zeros(batch, C, device=A.device)

    # We find k, the number of relevant pixels.
    k = int(top_t * H * W)

    for i in range(batch):

        # For every channel in every batch,
        for c in range(C):
            # Breakdown of this next line:
            # A is in form batch, channel, then a grid of HxW saliency scores acquired through the forward pass.
            # Thus, A[i, c] is just the grid.
            # .view(-1) here flattens the grid, then we find the top k values of that flattened grid.
            # In summary, this just finds the most important t% of the values in A[i, c], which is the set H+ in the paper.
            topk_vals, _ = torch.topk(A[i, c].view(-1), k)

            # Finally, this takes the mean of that set, as described in the paper again (p11, (3)).
            agg[i, c] = topk_vals.mean()
    return agg

# ------------------------ TRAINING ------------------------

# CUDA is not available to me, so I will simply use my CPU.
# Leaving this as a variable in case I find something else, though.
# Note: I didn't.
device = "cpu"
model = SaliencyModel().to(device)
# The second hyperparameter is the learning rate (lr).
optimizer = optim.Adam(model.parameters(), lr=(10**-3.5))

num_epochs = 100
num_batches = (len(train_dataset) // 12) + 1

# Set the best loss to infinity so it can decrease over time.
best_val_loss = float("inf")

# Don't worry, this is overwritten if a state dictionary is loaded of a previously trained model.
start_epoch = 0

# You will see these commented variables used later in a commented section of code.
# patience = 5
# early_stop_counter = 0

# If a checkpoint exists, we load it as shown and continue training it.
# I only check for "checkpoint.pth". This way, if I want to save a model somewhere and not continue training it, I can just change the file name.
if os.path.exists("checkpoint.pth"):
    print("Loading checkpoint...")
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", best_val_loss)

# Saves checkpoints for convenience of training.
# Since I'm only training on a CPU (an Intel i7-1355U at that), I have to stop training often.
# Also, this allows me to preserve good models, which really helps with finetuning hyperparameters.
def save_checkpoint(epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }, "checkpoint.pth")

# The final hyperparameter is beta, the regularization coefficient.
beta = 1e-3

# I pack all the training in a try-except to allow me to break at any time, as explained further later.
try:
    # This is the actual training loop the file alludes to.
    for epoch in range(start_epoch, num_epochs):

        model.train()
        train_loss = 0.0

        # This is a counter for printing updates on sets of 10 batches.
        # An update every epoch makes it take too long to check whether or not the code is running.
        i = 0

        for images, labels in train_loader:
            i += 1
            if i % 10 == 0:
                # Epochs are zero-indexed, but we print them one-indexed for convenience.
                print(f"Epoch: {epoch+1}. Batch: {i}/{num_batches}")

            images = images.to(device)
            labels = torch.tensor(
                # Finds the index of the 1 in the one-hot vectors through a list comprehension.
                [l.tolist().index(1) for l in labels],
                device=device
            )

            # Avoid gradient accumulation which completely messes up the model by zeroing all gradients before backpropagation.
            optimizer.zero_grad()

            # We run the forward pass, and
            h, A = model(images)
            # Find the predicted value.
            y_pred = aggregate_saliency(A)

            # We then calculate the CE loss on the prediction.
            loss_cls = criterion(y_pred, labels)

            # We impose the L1 regularization, as described in the paper (p11, (5)).
            loss_reg = A.abs().mean()
            # Then we use our hyperparameter beta, as described in the paper (p11, (6))
            loss = loss_cls + beta * loss_reg

            # Finally, we backpropagate, then apply learnings to the model.
            loss.backward()
            optimizer.step()

            # We add the losses over the entire training loop.
            train_loss += loss.item()

        # Finally, we divide by the number of batches to find the mean loss in training.
        train_loss /= len(train_loader)

        # We now add the validation step. This is crucial for avoiding overfitting.

        # Switch the model into evaluation mode.
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                # The logic here is very similar to that of the training loop.
                images = images.to(device)
                labels = torch.tensor(
                    [l.tolist().index(1) for l in labels],
                    device=device
                )

                h, A = model(images)
                y_pred = aggregate_saliency(A)

                loss_cls = criterion(y_pred, labels)
                loss_reg = A.abs().mean()
                loss = loss_cls + beta * loss_reg

                val_loss += loss.item()

        # Calculate the more practical valuation loss.
        val_loss /= len(val_loader)

        # We print the values so that the user understands when to stop the program.
        # Note that, at least on my computer, this runs quite slowly, so there is more than enough time to spot a good moment to stop the training.
        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Update the best valuation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
        # ----- EARLY STOPPING -----
        # This mechanism ended up not being very useful, so it is now commented.

        #     early_stop_counter = 0
        #     torch.save(model.state_dict(), "best_model.pth")
        # else:
        #     early_stop_counter += 1

        # save_checkpoint(epoch, val_loss)

        # if early_stop_counter >= patience:
        #     print("Early stopping triggered.")
        #     break

# We allow stopping half-way through training to avoid training through overfitting, or random crashes.
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving checkpoint...")
    # Note that this ignored error is literally impossible so long as num_epochs > 0, which it always will be.
    save_checkpoint(epoch) # pyright: ignore[reportPossiblyUnboundVariable]

except Exception:
    save_checkpoint(epoch) # pyright: ignore[reportPossiblyUnboundVariable]

# Finally, after exiting the loop if 100 epochs are reached, a checkpoint is saved anyways.
print("Training complete.")
save_checkpoint(epoch) # pyright: ignore[reportPossiblyUnboundVariable]