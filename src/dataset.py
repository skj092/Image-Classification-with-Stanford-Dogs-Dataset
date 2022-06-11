# dataset and dataloader class for image classification
from torch.utils.data import Dataset, DataLoader
import utils 
from PIL import Image
from torchvision import transforms
import torch

transform = transforms.Compose([transforms.ToTensor()])

class PetsDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).resize((224,224)).convert('RGB')
        label = utils.find_label(img_path)
        label = utils.class2id[label]

        if self.transform:  # apply transform to image
            img = self.transform(img)
        
        sample = {'image': img, 'label': torch.tensor(label)}

        return sample


if __name__=="__main__":
    dataset = Dataset(utils.train_images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, sample in enumerate(dataloader):
        print(sample['image'].shape)
        print(sample['label'])

