import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add the cloned AMD repository to the system path
sys.path.append('/content/AMD/models')
import modeling_finetune


def run_sanity_check():
    """
    This function verifies that the AMD model can overfit a single
    batch of random data, confirming that the training mechanics are functional.
    """
    print("🚀 Starting AMD Sanity Check...")

    # Use a GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Model Initialization ---
    num_classes = 10
    model = modeling_finetune.vit_small_patch16_224(
        num_classes=num_classes,
        drop_path_rate=0.1,
        head_drop_rate=0.0,
    )
    model.to(device)
    model.train()
    print("✅ Model instantiated successfully.")

    # --- 2. Create Dummy Data (with CORRECTED shape) ---
    # We create a batch with shape (N, D, C, H, W) and then permute it
    # to the required shape (N, C, D, H, W).
    batch_size = 2
    num_frames = 16
    
    # Create the tensor
    dummy_video_batch = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    # *** THIS IS THE FIX ***
    # Rearrange dimensions from (N, D, C, H, W) to (N, C, D, H, W)
    dummy_video_batch = dummy_video_batch.permute(0, 2, 1, 3, 4).to(device)
    
    # Create corresponding random labels
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    print(f"✅ Created a correctly shaped dummy video batch: {dummy_video_batch.shape}")

    # --- 3. Setup Optimizer and Loss Function ---
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    print("✅ Optimizer and Loss function are set up.")

    # --- 4. The Overfitting Loop ---
    print("\n--- Starting Overfitting Experiment (10 steps) ---")
    for step in range(10):
        optimizer.zero_grad()
        output = model(dummy_video_batch)
        loss = criterion(output, dummy_labels)
        loss.backward()
        optimizer.step()

        print(f"Step [{step+1:02d}/10], Loss: {loss.item():.4f}")

    print("--- Experiment Finished ---")

    # --- 5. Final Verification ---
    if loss.item() < 1.0:
        print("\n✅ SUCCESS: The loss decreased significantly, indicating the model is learning.")
    else:
        print("\n⚠️ WARNING: The loss did not decrease as expected. There might be an issue.")


if __name__ == "__main__":
    run_sanity_check()
