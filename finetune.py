import sys
import os
# Add the project's root directory to the Python path to find all local files
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from decord import VideoReader, cpu
import numpy as np
import argparse
from pathlib import Path

# Import necessary local files
import modeling_finetune
from random_erasing import RandomErasing # This will now be found

# --- Data Augmentation Class ---
class DataAugmentationForVideoMAE(object):
    def __init__(self, args, is_train=True):
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(self.input_mean, self.input_std)
        self.is_train = is_train
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.5, 1.0), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                normalize,
            ])

    def __call__(self, images):
        images = images.permute(1, 0, 2, 3) # T, C, H, W
        processed_frames = []
        for img_tensor in images:
            pil_img = transforms.ToPILImage()(img_tensor)
            processed_frames.append(self.transform(pil_img))
        
        return torch.stack(processed_frames, dim=1) # C, T, H, W

# --- KTH Dataset Class ---
class KTHDataset(Dataset):
    def __init__(self, root_path, annotation_file, transform=None):
        self.root_path = Path(root_path)
        self.transform = transform
        self.video_list = []
        with open(annotation_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.video_list.append((self.root_path / path, int(label)))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path, label = self.video_list[idx]
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            frame_indices = np.linspace(0, len(vr) - 1, 16, dtype=int)
            video = vr.get_batch(frame_indices).asnumpy()
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return torch.zeros((3, 16, 224, 224)), -1

        video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2) # C, T, H, W
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, label

# --- Main Fine-tuning Logic ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_train = DataAugmentationForVideoMAE(args)
    transform_val = DataAugmentationForVideoMAE(args, is_train=False)
    
    args.nb_classes = 6

    train_dataset = KTHDataset(root_path=args.data_path, annotation_file=os.path.join(args.data_path, "train.txt"), transform=transform_train)
    val_dataset = KTHDataset(root_path=args.data_path, annotation_file=os.path.join(args.data_path, "test.txt"), transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = modeling_finetune.vit_small_patch16_224(num_classes=args.nb_classes)
    
    checkpoint = torch.load(args.finetune, map_location='cpu')
    print("Loading pre-trained checkpoint from:", args.finetune)
    
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint
    
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint.")
            del checkpoint_model[k]
            
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Start training for {args.epochs} epochs")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for videos, labels in train_loader:
            if -1 in labels: continue
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                if -1 in labels: continue
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if total > 0:
            accuracy = 100 * correct / total
            print(f"Validation Accuracy: {accuracy:.2f}%")

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(model.state_dict(), "kth_best_model.pth")
                print(f"New best model saved with accuracy: {best_acc:.2f}%")
        else:
            print("Validation set is empty. No accuracy calculated.")
            
    print(f"Finished Training. Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('KTH fine-tuning script', add_help=False)
    parser.add_argument('--finetune', default='/content/vits_k400_finetune_150e.pth', type=str)
    parser.add_argument('--data_path', default='dataset/KTH', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--color_jitter', type=float, default=None)
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    
    args = parser.parse_args(args=[])
    main(args)
