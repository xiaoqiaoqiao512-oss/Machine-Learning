from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_dataloader(batch_size = 64):

    train_dataset = datasets.MNIST(
        root = "./data",
        train = True,
        download = True,
        transform = ToTensor()
    )

    test_dataset = datasets.MNIST(
        root = "./data",
        train = False,
        download = True,
        transform = ToTensor()
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False
    )

    return train_dataloader, test_dataloader