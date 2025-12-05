from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        import gzip
        import struct
        
        # 解析图像文件
        with gzip.open(image_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            X = np.frombuffer(f.read(), dtype=np.uint8)
            X = X.reshape(num_images, rows, cols, 1)  # (N, H, W, C)
            X = X.astype(np.float32) / 255.0
        
        # 解析标签文件
        with gzip.open(label_filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            y = np.frombuffer(f.read(), dtype=np.uint8)
        
        self.images = X
        self.labels = y
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        label = self.labels[index]
        
        # 应用变换
        if self.transforms is not None:
            for transform in self.transforms:
                img = transform(img)
        
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION