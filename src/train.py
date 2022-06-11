from models import get_model
import utils 
from dataset import transform, PetsDataset
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch.optim as optim
import torch
from engine import train, evaluate, predict


if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_train = PetsDataset(utils.train_images, transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    dataset_valid = PetsDataset(utils.valid_images, transform=transform)
    dataloader_valid = DataLoader(dataset_valid, batch_size=32, shuffle=True)

    dataset_test = PetsDataset(utils.test_images, transform=transform)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

    model = get_model(pretrained=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(1):
        train(dataloader_train, model, optimizer, device)
        val_loss = evaluate(dataloader_valid, model, device)
        print(f"Epoch: {epoch}, Val Loss: {val_loss}")
        torch.save(model.state_dict(), f"../models/model_{epoch}.pt")



