# dataset and dataloader class for image classification
from torch.utils.data import Dataset, DataLoader
import utils 
from PIL import Image


class Dataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        label = utils.find_label(img_path)

        sample = {'image': img, 'label': label}

        if self.transform:
            imgs = self.transform(sample)

        return sample


if __name__=="__main__":
    dataset = Dataset(utils.train_images)
    print(dataset[0])

