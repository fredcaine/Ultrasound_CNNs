import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from matplotlib.widgets import Button

# This is the valuation transform, not the training transform, as there is no reason to augment the evaluation files.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class SaliencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)

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

        self.saliency_conv = nn.Conv2d(512, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.features(x)
        A = self.sigmoid(self.saliency_conv(h))
        return h, A

def aggregate_saliency(A, top_t=0.25):
    batch, C, H, W = A.shape
    agg = torch.zeros(batch, C, device=A.device)
    k = int(top_t * H * W)

    for i in range(batch):
        for c in range(C):
            topk_vals, _ = torch.topk(A[i, c].view(-1), k)
            agg[i, c] = topk_vals.mean()

    return agg

device = "cpu"
model = SaliencyModel().to(device)

checkpoint = torch.load("saliency_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

base_dir = "Images"

# We define key variables.
correct = 0
total = 0
all_preds = []
all_labels = []

shown = 0
# The maximum number of images that show
max_shown = 500

# Going through the folder's architecture, first into the correct subdirectory.
for label_idx, folder in enumerate(["normal", "benign", "malignant"]):
    folder_path = os.path.join(base_dir, folder)

    # Then iterating through this subdirectory.
    for idx, fname in enumerate(os.listdir(folder_path)):
        path = os.path.join(folder_path, fname)

        img = Image.open(path).convert("L")
        img = transform(img).unsqueeze(0).to(device)

        # Run the forward pass.
        # torch.no_grad() is good for evaluation as it avoids memory usage on potential gradient descent being required.
        with torch.no_grad():

            h, A = model(img)

            # For making the figure set, it was easier if I only had to check through saliency maps for specific classes.
            # So, this if statement allows it to only display the maps if the aggregate is whichever class you choose.
            # Replace the '3' with:
            # 0 for normal images only,
            # 1 for benign images only, or
            # 2 for malignant images only. (You can also replace == with 'in' and a tuple of numbers if you would like multiple classes to show up).
            if shown < max_shown and torch.argmax(aggregate_saliency(A), dim=1).item() == 3:

                # Aggregate the prediction to know what class it's sorted into.
                pred = torch.argmax(aggregate_saliency(A), dim=1).item()

                # Converts from [1,3,256,256] to numpy [256,256,3] and transfers to CPU so libraries can work with it.
                img_np = img.squeeze(0).cpu().permute(1, 2, 0).numpy()

                # Converts from [1,3,8,8] to [3,8,8] and transfers to CPU so numpy can work with it.
                saliency = A.squeeze(0).cpu()  # [3, H, W]

                colors = [
                    np.array([0, 1, 0]),  # class 0
                    np.array([0, 0, 1]),  # class 1
                    np.array([1, 0, 0])   # class 2
                ]

                overlay = np.zeros((256, 256, 3))
                alpha_total = np.zeros((256, 256))

                for c in range(3):

                    # Get the map of this channel with [1,1,8,8] so torch can work with it.
                    saliency_map = saliency[c].unsqueeze(0).unsqueeze(0)

                    # Solely for aesthetics, we upscale the map to 256×256.
                    saliency_map = nn.functional.interpolate(
                        saliency_map,
                        size=(256, 256),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze().numpy()

                    # Normalise all values to be within 0--1.
                    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

                    # The one-hot arrays are in the RGB format.
                    if c == 2:
                        colour = np.array([1, 0, 0])
                    elif c == 1:
                        colour = np.array([0, 0, 1])
                    else:
                        colour = np.array([0, 1, 0])

                    # Ensure nothing goes too opaque so the image is visible (maximum alpha is 0.6**2.5 * 0.8 ~= 0.223)
                    alpha = np.clip(saliency_map - 0.4, 0, 1)

                    # These are just random numbers to try to boost the difference between high and low attention areas.
                    alpha = 0.8 * alpha ** 2.5

                    overlay += alpha[..., None] * colour
                    alpha_total += alpha

                alpha_total = np.clip(alpha_total, 0, 1)

                # Overlay the image and the overlay we just created.
                composite = img_np * (1 - alpha_total[..., None]) + overlay
                composite = np.clip(composite, 0, 1)

                # Show it with matplotlib.
                fig, ax = plt.subplots(figsize=(4, 4))
                plt.subplots_adjust(bottom=0.15)
                ax.imshow(composite)
                ax.axis("off")
                ax.set_title("Saliency Overlay")

                # Creating and placing a button to save images immediately (for the figure set).
                save_ax = plt.axes([0.72, 0.02, 0.25, 0.08])
                save_button = Button(save_ax, "Save")

                def save(event):
                    fname = f"saliency_pred_{pred}_idx_{shown}.png"
                    plt.imsave(fname, composite)
                    print(f"saved -> {fname}")

                save_button.on_clicked(save)

                plt.show()

                shown += 1

            y_pred = aggregate_saliency(A)

            # For AUROC.
            probs = torch.softmax(y_pred, dim=1).squeeze(0).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(label_idx)

            # We find the most likely prediction.
            pred = torch.argmax(y_pred, dim=1).item()

            if pred == label_idx:
                correct += 1

        total += 1

accuracy = correct / total
print(f"Accuracy: {round(100 * accuracy, 2)}%.")

# Transforms into one-hot vectors.
y_true_bin = label_binarize(np.array(all_labels), classes=[0,1,2])

# Plotting the graph.
plt.figure(figsize=(6,6))
fpr, tpr, thresholds = roc_curve(y_true_bin[:,0], np.array(all_preds)[:,0])
plt.plot(fpr, tpr, "g:", label=f"Normal (AUROC = {round(auc(fpr, tpr), 3)})")
fpr, tpr, thresholds = roc_curve(y_true_bin[:,1], np.array(all_preds)[:,1])
plt.plot(fpr, tpr, "b--", label=f"Benign (AUROC = {round(auc(fpr, tpr), 3)})")
fpr, tpr, thresholds = roc_curve(y_true_bin[:,2], np.array(all_preds)[:,2])
plt.plot(fpr, tpr, "r-", label=f"Malignant (AUROC = {round(auc(fpr, tpr), 3)})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.axis("square")
plt.legend()
# Uncomment this if you would like to save the graph.
# plt.savefig("ROC Curve.svg", format="svg", bbox_inches="tight")
plt.show()

# This is a multiclass AUROC.
auroc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
print(f"AUROC: {round(auroc, 4)}")

# Output:
# Accuracy: 92.05%.
# AUROC: 0.972
# Pretty good.
