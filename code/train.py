import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFile
from pathlib import Path
import random
import numpy as np
import os
import csv

# ============================================================================
# CONFIGURATION
# ============================================================================

EPOCHS = 12  # reduced for faster experiments
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
RANDOM_SEED = 42
TRAIN_DIR = Path(__file__).parent / "data" / "train"
VAL_DIR = Path(__file__).parent / "data" / "val"
MODEL_PATH = Path(__file__).parent / "best_model.pth"
CLASS_NAMES = ["chihuahua", "muffin"]
LABEL_MAP = {"chihuahua": 0, "muffin": 1}
NUM_CLASSES = len(CLASS_NAMES)
LABEL_SMOOTHING = 0.06
CUTMIX_PROB = 0.5
PSEUDO_LABEL_THRESHOLD = 0.70
UNLABELED_DIR = Path(__file__).parent / "data" / "train" / "undefined"
CORRECTED_LABELS_PATH = Path(__file__).parent / "corrected_samples.txt"

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("ResNet-18: random init (no pretrained weights — competition rules)")


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"[OK] Random seed set to {seed}")


# ============================================================================
# MODEL
# ============================================================================

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)


# ============================================================================
# TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize(160, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomResizedCrop(140, scale=(0.65, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.22, hue=0.08),
        transforms.RandomGrayscale(p=0.1),
    ], p=0.6),
    transforms.RandomAffine(degrees=12, shear=8, scale=(0.88, 1.12)),
    transforms.RandomRotation(12),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
])

val_transform = transforms.Compose([
    transforms.Resize(160, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(140),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def rand_bbox(size, lam):
    _, _, H, W = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    x_shuffled = x[index]
    y_a, y_b = y, y[index]
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x[:, :, y1:y2, x1:x2] = x_shuffled[:, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / float(x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabeledImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def collect_labeled_samples(root_dirs):
    samples = []
    for root in root_dirs:
        for class_name, label in LABEL_MAP.items():
            folder = Path(root) / class_name
            if not folder.exists():
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in sorted(folder.glob(ext)):
                    samples.append((img_path, label))
    return samples


def create_train_val_split(samples, val_fraction=0.1, seed=RANDOM_SEED):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    indices_by_label = {label: [] for label in LABEL_MAP.values()}
    for idx, (_, label) in enumerate(samples):
        indices_by_label[label].append(idx)

    train_indices = []
    val_indices = []
    rng = random.Random(seed)
    for label, indices in indices_by_label.items():
        rng.shuffle(indices)
        val_size = max(1, int(len(indices) * val_fraction))
        val_indices.extend(indices[:val_size])
        train_indices.extend(indices[val_size:])

    if len(train_indices) == 0:
        train_indices = val_indices[:-1]
        val_indices = val_indices[-1:]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def collect_unlabeled_images(unlabeled_dir):
    images = []
    if unlabeled_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in sorted(unlabeled_dir.glob(ext)):
                images.append(img_path)
    return images


def load_corrected_samples(path, unlabeled_dir=UNLABELED_DIR):
    samples = []
    if not path.exists():
        return samples

    current_image = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.endswith(('.jpg', '.jpeg', '.png')) and not stripped.startswith('Original Label:'):
                current_image = stripped.split()[-1]
            elif stripped.startswith("Corrected Label:") and current_image is not None:
                label_text = stripped.split(":", 1)[1].strip().lower()
                if label_text in LABEL_MAP:
                    image_path = Path(unlabeled_dir) / current_image
                    if image_path.exists():
                        samples.append((image_path, LABEL_MAP[label_text]))
                    current_image = None
    return samples


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path


def build_loaders(train_samples, val_samples=None, use_sampler=True):
    if len(train_samples) == 0:
        raise RuntimeError("No training samples found.")

    train_counts = {label: 0 for label in LABEL_MAP.values()}
    for _, label in train_samples:
        train_counts[label] += 1

    print(f"Found {len(train_samples)} training images")
    print("Training label counts:")
    for class_name, label in LABEL_MAP.items():
        print(f"  {class_name}: {train_counts[label]}")

    train_dataset = LabeledImageDataset(train_samples, transform=train_transform)
    if use_sampler:
        train_weights = [1.0 / train_counts[label] for _, label in train_samples]
        train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, pin_memory=torch.cuda.is_available())
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    val_loader = None
    if val_samples is not None:
        if len(val_samples) == 0:
            raise RuntimeError("No validation samples found.")
        print(f"Found {len(val_samples)} validation images")
        val_counts = {label: 0 for label in LABEL_MAP.values()}
        for _, label in val_samples:
            val_counts[label] += 1
        print("Validation label counts:")
        for class_name, label in LABEL_MAP.items():
            print(f"  {class_name}: {val_counts[label]}")
        val_dataset = LabeledImageDataset(val_samples, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader


def get_teacher_predictions(model, unlabeled_loader, threshold=PSEUDO_LABEL_THRESHOLD):
    model.eval()
    pseudo_samples = []
    pseudo_counts = {label: 0 for label in LABEL_MAP.values()}
    with torch.no_grad():
        for images, paths in tqdm(unlabeled_loader, desc="Pseudo-labeling"):
            images = images.to(device)
            probs = predict_with_tta_batch(model, images)
            confidence, preds = probs.max(dim=1)
            for path, pred, conf in zip(paths, preds, confidence):
                if conf.item() >= threshold:
                    pseudo_samples.append((Path(path), int(pred.item())))
                    pseudo_counts[int(pred.item())] += 1

    print("Pseudo-label distribution:")
    for class_name, label in LABEL_MAP.items():
        print(f"  {class_name}: {pseudo_counts[label]}")
    return pseudo_samples


def predict_with_tta_batch(model, images):
    variants = [
        images,
        torch.flip(images, dims=[3]),
    ]
    probs = []
    with torch.no_grad():
        for variant in variants:
            logits = model(variant.to(device))
            probs.append(F.softmax(logits, dim=1))
    return torch.stack(probs).mean(0)


def train_model(train_samples, val_samples=None, model_save_path=None, use_sampler=True, early_stopping=True):
    train_loader, val_loader = build_loaders(train_samples, val_samples, use_sampler=use_sampler)
    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_accuracy = 0.0
    best_model_state = None
    patience = 6
    patience_counter = 0

    print("\n" + "=" * 60)
    print("  Starting Training")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if np.random.rand() < CUTMIX_PROB:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            else:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)

            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    pred = model(images).argmax(1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
            val_accuracy = 100.0 * val_correct / val_total
            print(f"    Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {val_accuracy:.2f}%")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print("    --> New best model!")
            else:
                patience_counter += 1
                if early_stopping and patience_counter >= patience:
                    print(f"    --> Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"    Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    if best_model_state is not None and val_loader is not None:
        model.load_state_dict(best_model_state)
    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
        print(f"    Saved {model_save_path}")

    print("\n" + "=" * 60)
    if val_loader is not None:
        print(f"  Best validation accuracy: {best_val_accuracy:.2f}%")
    else:
        print("  Training complete (no validation set)")
    print("=" * 60)
    return model, best_val_accuracy


def train():
    set_seed(RANDOM_SEED)
    labeled_samples = collect_labeled_samples([TRAIN_DIR, VAL_DIR])
    corrected_samples = load_corrected_samples(CORRECTED_LABELS_PATH)
    if corrected_samples:
        print(f"Loaded {len(corrected_samples)} corrected samples from {CORRECTED_LABELS_PATH}")
        labeled_samples.extend(corrected_samples)

    if len(labeled_samples) == 0:
        raise RuntimeError("No labeled samples found in train/val or corrected labels.")

    print(f"Found {len(labeled_samples)} labeled samples across train/val + corrected labels")

    train_indices, val_indices = create_train_val_split(labeled_samples, val_fraction=0.1)
    teacher_train_samples = [labeled_samples[i] for i in train_indices]
    teacher_val_samples = [labeled_samples[i] for i in val_indices]

    print("\n[1/3] Training teacher model using labeled data")
    teacher_model, teacher_acc = train_model(teacher_train_samples, teacher_val_samples, model_save_path=None)

    unlabeled_paths = collect_unlabeled_images(UNLABELED_DIR)
    corrected_names = {p.name for p, _ in corrected_samples}
    unlabeled_paths = [p for p in unlabeled_paths if p.name not in corrected_names]
    print(f"Found {len(unlabeled_paths)} unlabeled images available for pseudo-labeling")

    pseudo_samples = []
    if unlabeled_paths:
        unlabeled_dataset = UnlabeledImageDataset(unlabeled_paths, transform=val_transform)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        pseudo_samples = get_teacher_predictions(teacher_model, unlabeled_loader, threshold=PSEUDO_LABEL_THRESHOLD)
        print(f"Selected {len(pseudo_samples)} pseudo-labeled images with threshold {PSEUDO_LABEL_THRESHOLD}")

    print("\n[2/3] Training final model on labeled + pseudo-labeled data with validation holdout")
    final_train_samples = teacher_train_samples + pseudo_samples
    final_model, final_acc = train_model(final_train_samples, val_samples=teacher_val_samples, model_save_path=MODEL_PATH, use_sampler=True, early_stopping=True)

    if pseudo_samples:
        save_pseudo_labels(pseudo_samples)

    print("\n[3/3] Final model training complete")
    return final_model


def save_pseudo_labels(pseudo_samples, output_path=Path(__file__).parent / "pseudo_labels.csv"):
    if not pseudo_samples:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        for img_path, label in pseudo_samples:
            writer.writerow([str(img_path), label])
    print(f"Saved {len(pseudo_samples)} pseudo labels to {output_path}")


if __name__ == "__main__":
    train()
