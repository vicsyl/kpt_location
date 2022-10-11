import matplotlib.pyplot as plt


def show_np(img, title):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.show()
    plt.close()


def show_torch(img, title):
    show_np(img.numpy(), title)