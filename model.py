import torch
import torch.nn as nn
import torch.nn.functional as F

# A small CNN for classifying leaf images into 38 classes.
# Kept simple on purpose - the project is about testing the entropy-gating
# mechanism, not about squeezing out maximum possible accuracy.

NUM_CLASSES = 38
IMAGE_SIZE = 64  # we will resize all images to 64x64 before feeding the model


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()

        # Convolutional layers: extract visual features (edges, textures, spots on leaves)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 3 pooling layers, a 64x64 image becomes 8x8
        # (64 -> 32 -> 16 -> 8), with 64 channels at that point.
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 16 x 32 x 32
        x = self.pool(F.relu(self.conv2(x)))  # -> 32 x 16 x 16
        x = self.pool(F.relu(self.conv3(x)))  # -> 64 x 8 x 8

        x = x.view(x.size(0), -1)  # flatten into a single vector per image

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # raw scores (logits) for each of the 38 classes

        return x


if __name__ == "__main__":
    # Quick sanity check: create the model and pass in a fake batch of images
    # to confirm the shapes all line up correctly.
    model = SimpleCNN()
    fake_batch = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)  # 4 fake images
    output = model(fake_batch)

    print(f"Model created successfully.")
    print(f"Input shape: {fake_batch.shape}")
    print(f"Output shape: {output.shape}")  # should be (4, 38)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")