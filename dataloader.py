
import os
import cv2
from paddle.io import Dataset, DataLoader
import numpy as np

def resizeImage(image, size, label=None):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if label is not None:
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
        return image, label
    return image

class CelebA(Dataset):
    def __init__(self, datadir, img_size, mode='train'):
        super(CelebA, self).__init__()
        self.data_dir = datadir
        self.image_size = img_size
        assert mode in ['train', 'val']
        self.mode = mode
        file = os.path.join(datadir, mode) + '.txt'
        f = open(file, 'r')
        self.files = f.readlines()
        self.num_file = len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx].strip('\n')  # 去掉列表中每一个元素的换行符
        img_path = os.path.join(self.data_dir, "CelebA-HQ-img", file_path)
        img_path = img_path + ".jpg"
        annotation_path = os.path.join(self.data_dir, "annotation", file_path)
        annotation_path = annotation_path + ".png"
        image = cv2.imread(img_path)
        label = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        image, label = resizeImage(image, size=self.image_size, label=label)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = label[:, :, np.newaxis]

        return image/255.0, label

    def __len__(self):
        return self.num_file

def main():
    batch_size = 2
    train_dataset = CelebA('dataset/CelebAMask-HQ/', 512, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    num_epoch = 2
    for epoch in range(1, num_epoch + 1):
        print(f'Epoch [{epoch}/{num_epoch}]:')
        for idx, (data, label) in enumerate(train_loader):
            print(f'Iter {idx}, Data shape: {data.shape}, Label shape: {label.shape}')
            if idx ==10:
                break

if __name__ == "__main__":
    main()