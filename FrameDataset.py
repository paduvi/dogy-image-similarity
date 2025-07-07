from torchvision import transforms
from torch.utils.data import Dataset


# --- Dataset để batch các khung hình ---
class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.transform(self.frames[idx])
