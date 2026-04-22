# Fredrick Farouk. evaluation_BUS-BUSI.py
# This is similiar to the other evaluation file, but using the same dataset as training (linked in saliency_training.py).
# I talked more about this in README.md.

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.metrics import roc_auc_score

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
            y_pred = aggregate_saliency(A)

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

auroc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
print(f"AUROC: {round(auroc, 4)}")

# This is the output:
# Accuracy: 92.05%
# AUROC: 0.972

# Pretty good.
