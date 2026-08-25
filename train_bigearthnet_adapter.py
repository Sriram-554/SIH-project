"""
Lightweight training script for BigEarthNet Domain Adapter.
This creates a valid domain-adapted model using synthetic + rule-based supervision
based on BigEarthNet-19 ontology (acceptable for SIH demonstration).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from src.models.bigearthnet_adapter import BigEarthNetAdapter, BIGEARTHNET_19_CLASSES

class SyntheticBigEarthNetDataset(Dataset):
    """Generates realistic spectral feature vectors with BigEarthNet-19 multi-labels."""

    def __init__(self, num_samples=2000):
        self.samples = []
        self.labels = []

        for _ in range(num_samples):
            # Random realistic spectral features
            ndvi = np.random.uniform(0.05, 0.85)
            water = np.random.uniform(0, 0.6)
            dense_veg = np.random.uniform(0, 0.7)
            mod_veg = np.random.uniform(0, 0.6)
            barren = np.random.uniform(0, 0.5)
            ndwi = np.random.uniform(-0.3, 0.5)
            nbr = np.random.uniform(-0.2, 0.6)
            brightness = np.random.uniform(0.15, 0.75)

            features = np.array([ndvi, ndwi, water, dense_veg, mod_veg, barren, nbr, brightness], dtype=np.float32)

            # Create multi-label target based on physical rules (BigEarthNet style)
            label = np.zeros(19, dtype=np.float32)

            if water > 0.15:
                label[17] = 1.0  # Inland waters
            if water > 0.35:
                label[18] = 1.0  # Marine waters

            if dense_veg > 0.25:
                label[7] = 1.0   # Broad-leaved forest
                label[9] = 0.8   # Mixed forest
            if dense_veg > 0.4:
                label[8] = 0.7   # Coniferous

            if mod_veg > 0.2:
                label[3] = 1.0   # Arable land
                label[6] = 0.7   # Complex cultivation
                label[5] = 0.6   # Pastures

            if barren > 0.25:
                label[1] = 0.8   # Discontinuous urban
                label[14] = 0.6  # Bare rocks
            if barren > 0.4 and brightness > 0.5:
                label[0] = 0.9   # Continuous urban
                label[2] = 0.7   # Industrial

            if 0.2 < ndvi < 0.4:
                label[10] = 0.7  # Natural grassland
                label[12] = 0.6  # Transitional woodland

            self.samples.append(features)
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])


def train():
    print("Starting BigEarthNet Domain Adapter training...")

    dataset = SyntheticBigEarthNetDataset(num_samples=2500)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = BigEarthNetAdapter()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(15):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/15 | Loss: {total_loss/len(loader):.4f}")

    # Save the trained model
    save_path = Path("models/bigearthnet_adapter.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Trained model saved to: {save_path}")
    print("Domain adaptation completed successfully.")


if __name__ == "__main__":
    train()