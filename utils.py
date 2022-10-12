import matplotlib.pyplot as plt
import torch


def show_np(img, title):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.show()
    plt.close()


def show_torch(img, title):
    show_np(img.numpy(), title)


def avg_loss(data):
    return (torch.linalg.norm(data, dim=1) ** 2).sum() / data.shape[0]
