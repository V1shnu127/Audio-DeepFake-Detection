#!/bin/env/python3

import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

# -------------------------
# 1. Dataset Class for Raw Audio
# -------------------------
class VoiceDataset(Dataset):
    def __init__(self, data_dir, label, target_length=16000):  
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.label = label  
        self.target_length = target_length

    def __len__(self):
        return len(self.files) * 2  # Each file gives two samples

    def __getitem__(self, idx):
        file_idx = idx // 2  
        part = idx % 2  # 0 = first second, 1 = second second
        
        file_path = os.path.join(self.data_dir, self.files[file_idx])
        y, sr = librosa.load(file_path, sr=16000)  

        # Normalize the waveform
        y = y / np.max(np.abs(y))

        # Extract 1-second segments
        y = y[part * 16000 : (part + 1) * 16000]

        return torch.tensor(y.reshape(1, 1, self.target_length), dtype=torch.float32), torch.tensor(self.label, dtype=torch.long)


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.PReLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        device="cuda"  # Default to GPU
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1, device=self.device) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features).to(self.device))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order).to(self.device)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation().to(self.device)
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.base_weight) 
        torch.nn.init.constant_(self.spline_weight, 0.01)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features, device=self.device)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                self.scale_spline
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.grid.to(x.device)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1).to(self.device)
        B = y.transpose(0, 1).to(self.device)

        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        
        return base_output + spline_output

class RawAudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(RawAudioCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2) 
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        

        # Compute the output size after convolutions + pooling
        conv_output_size = 64 * (16000 // 8)  # Assuming 16kHz input

        # Replace final FC layers with KANLinear
        self.kan_fc1 = KANLinear(conv_output_size, 128).to("cuda")
        self.kan_fc2 = KANLinear(128, num_classes).to("cuda")

    def forward(self, x):
        x = x.squeeze(2).to("cuda")  # Ensure input is on GPU -> [batch_size, 1, 16000]

        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.selu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten before FC layers
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        x = self.kan_fc1(x)  
        x = self.kan_fc2(x)
        return x


# -------------------------
# 3. Function to Compute Accuracy
# -------------------------
def compute_accuracy(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation (faster)
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Get predicted class (0 for real, 1 for fake)
            _, predicted = torch.max(outputs, 1)

            # Compare with actual labels
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total  # Accuracy as a fraction (0-1)

import os
import random
import shutil

# Root directory (Update this path as per your Kaggle dataset)
dataset_root = "/kaggle/input/for-2sec-dataset/dataset"  # Adjust if needed

# Define dataset splits and categories
splits = ["training", "validation", "testing"]
categories = ["fake", "real"]

# Collect all .wav files
all_files = {cat: [] for cat in categories}

for split in splits:
    for category in categories:
        folder_path = os.path.join(dataset_root, split, category)
        if os.path.exists(folder_path):
            all_files[category] += [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

# Shuffle the dataset
for category in categories:
    random.shuffle(all_files[category])

# Define new split ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

# Function to split data
def split_data(file_list, train_ratio, val_ratio):
    total_files = len(file_list)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    train_files = file_list[:train_count]
    val_files = file_list[train_count:train_count + val_count]
    test_files = file_list[train_count + val_count:]
    
    return train_files, val_files, test_files

# Apply split to shuffled files
new_splits = {split: {cat: [] for cat in categories} for split in splits}
for category in categories:
    new_splits["training"][category], new_splits["validation"][category], new_splits["testing"][category] = split_data(all_files[category], train_ratio, val_ratio)

# Create new dataset structure
shuffled_root = "/kaggle/working/shuffled_dataset"
for split in splits:
    for category in categories:
        new_folder = os.path.join(shuffled_root, split, category)
        os.makedirs(new_folder, exist_ok=True)

        # Move shuffled files
        for file_path in new_splits[split][category]:
            shutil.copy(file_path, os.path.join(new_folder, os.path.basename(file_path)))  # Use copy to keep original dataset

print("Dataset shuffled and saved to:", shuffled_root)

# -------------------------
# 4. Load Datasets
# -------------------------
train_dataset = VoiceDataset("/kaggle/working/shuffled_dataset/training/real", label=0) + VoiceDataset("/kaggle/working/shuffled_dataset/training/fake", label=1)
val_dataset = VoiceDataset("/kaggle/working/shuffled_dataset/validation/real", label=0) + VoiceDataset("/kaggle/working/shuffled_dataset/validation/fake", label=1)
test_dataset = VoiceDataset("/kaggle/working/shuffled_dataset/testing/real", label=0) + VoiceDataset("/kaggle/working/shuffled_dataset/testing/fake", label=1)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------
# 5. Train the Model
# -------------------------

# Initialize best accuracy
best_val_acc = 0.0  
best_model_path = "best_model.pth"  # File to save the best model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RawAudioCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

num_epochs = 20
for epoch in range(num_epochs):  
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track training accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

    
    train_acc = correct / total

    # Validation phase
    model.eval()  
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():  # No need to track gradients for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_acc = correct_val / total_val  # Compute validation accuracy

    # Check if the current validation accuracy is the best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved at epoch {epoch+1} with val_acc: {best_val_acc:.4f}")

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")

# Load the best model after training
model.load_state_dict(torch.load(best_model_path))
print("🚀 Loaded the best model with highest validation accuracy!")

# -------------------------
# 6. Compute Test Accuracy
# -------------------------
test_acc = compute_accuracy(model, test_loader, device)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

def compute_eer(label, pred):
    fpr, tpr, threshold = roc_curve(label, pred)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_1 + eer_2) / 2
    return eer

# Assuming model and test_loader are defined
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Compute confusion matrix
conf_mat = confusion_matrix(all_labels, all_preds)
sns.heatmap(conf_mat, cmap="flare", annot=True, fmt="g",
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("ConfusionMatrix_KAN_CNN.png")
plt.show()

# Compute F1 Score
f1 = f1_score(all_labels, all_preds, average='macro')
print('F1 Score:', f1)

# Compute EER
eer = compute_eer(all_labels, all_preds)
print('Equal Error Rate (EER): {:.3f}'.format(eer))
