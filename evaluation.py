# Fredrick Farouk. evaluation.py
# This file runs the model's forward pass on every image in the dataset to evaluate its accuracy.
# There will be no comments on lines that are identical to lines in other files.
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

# This is the valuation transform, not the training transform, as there is no reason to augment the evaluation files.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

class SaliencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

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
    k = max(1, int(top_t * H * W))

    for i in range(batch):
        for c in range(C):
            topk_vals, _ = torch.topk(A[i, c].reshape(-1), k)
            agg[i, c] = topk_vals.mean()

    return agg

device = "cpu"
model = SaliencyModel().to(device)

checkpoint = torch.load("saliency_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

base_dir = Path("BUS-UCLM/Images")
csv_path = Path("BUS-UCLM/image_level_information.csv")

# Make sure this matches the class order used when the model was trained.
class_to_idx = {
    "Normal": 0,
    "Benign": 1,
    "Malignant": 2
}

# We load y_true labels from the CSV file.
df = pd.read_csv(csv_path)
df["Image"] = df["Image"].astype(str)
df["Label"] = df["Label"].astype(str)

y_true_map = dict(zip(df["Image"], df["Label"]))

# Going through the folder's architecture recursively, first into the correct subdirectory.
# This will visit every subfolder of every subfolder, which is what we need.
correct = 0
total = 0

y_true_list = []
y_score_list = []

for img_path in base_dir.rglob("*.png"):
    fname = img_path.name

    true_label = class_to_idx[y_true_map[fname]]

    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)

    # Run the forward pass.
    # torch.no_grad() is good for evaluation as it avoids memory usage on potential gradient descent being required.
    with torch.no_grad():
        h, A = model(img)
        y_pred = aggregate_saliency(A)
        # Normalizing because of the annoying error for calculating AUROC.
        y_pred = y_pred / y_pred.sum(dim=1, keepdim=True)

    # We find the most likely prediction.
    pred = torch.argmax(y_pred, dim=1).item()

    if pred == true_label:
        correct += 1

    total += 1

    y_true_list.append(true_label)
    y_score_list.append(y_pred.squeeze(0).cpu().numpy())

accuracy = correct / total if total > 0 else 0.0
print(f"Accuracy: {round(100 * accuracy, 2)}%.")

y_true_tensor = torch.tensor(y_true_list)
y_score_tensor = torch.tensor(y_score_list)

auroc = roc_auc_score(y_true_tensor.numpy(), y_score_tensor.numpy(), multi_class="ovr")
print(f"AUROC: {round(auroc, 4)}")

# Output:
# Accuracy: 59.15%.
# AUROC: 0.7404
# Not bad.
