import os
from pathlib import Path

def generate_kth_annotations(dataset_path: str):
    """
    Generates train.txt and test.txt for the KTH dataset.
    
    The standard split uses persons 1-16 for training, 17-25 for testing.
    The script assumes the dataset is organized into subfolders by class name.
    """
    print(f"Scanning dataset path: {dataset_path}")
    dataset_root = Path(dataset_path)
    if not dataset_root.is_dir():
        print(f"Error: Dataset directory not found at {dataset_path}")
        return

    actions = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    label_map = {action: i for i, action in enumerate(actions)}
    
    print(f"Found {len(actions)} actions: {actions}")

    train_files = []
    test_files = []

    # Per the KTH website, the split is based on the person's ID
    train_ids = {f"person{i:02d}" for i in range(1, 17)} # Persons 1-16
    test_ids = {f"person{i:02d}" for i in range(17, 26)} # Persons 17-25

    for action_dir in dataset_root.iterdir():
        if not action_dir.is_dir():
            continue
        
        action_name = action_dir.name
        label = label_map[action_name]
        
        for video_file in action_dir.glob("*.avi"):
            person_id = video_file.name.split('_')[0]
            relative_path = video_file.relative_to(dataset_root)
            
            entry = f"{relative_path} {label}\n"
            
            if person_id in train_ids:
                train_files.append(entry)
            elif person_id in test_ids:
                test_files.append(entry)

    # Write the annotation files to the dataset directory
    with open(dataset_root / "train.txt", "w") as f:
        f.writelines(sorted(train_files))
    
    with open(dataset_root / "test.txt", "w") as f:
        f.writelines(sorted(test_files))
        
    print(f"Successfully created annotation files.")
    print(f"Training samples: {len(train_files)}")
    print(f"Testing samples: {len(test_files)}")


if __name__ == "__main__":
    # The path should point to the folder containing action subdirectories
    kth_data_path = "/content/AMD/dataset/KTH"
    generate_kth_annotations(kth_data_path)
