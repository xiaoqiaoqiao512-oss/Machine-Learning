import torch
import torch.nn as nn

from dataset import get_dataloader
from model import MLP

def train(device):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    train_dataloader, test_dataloader = get_dataloader(batch_size=64)

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 0.01
    )

    epoches = 5

    for epoch in range(epoches):
        model.train()
        total_loss = 0

        for images,labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)

            loss = criterion(pred,labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(epoch+1,total_loss/len(train_dataloader))

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images,labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                pred = model(images)

                predict = pred.argmax(dim=1)

                correct += (predict == labels).sum().item()

                total += labels.size(0)
        print("Accuracy: {:.2f}%".format(correct/total*100))
        # torch.save(model.state_dict(),"mnist.pth")

if __name__ == "__main__":
    train(torch.device)