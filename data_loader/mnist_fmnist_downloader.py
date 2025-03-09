import torchvision

path_name = "."

# Download MNIST
torchvision.datasets.MNIST(
    root = path_name + '/mnist', download=True
)

# Download FashionMNIST
torchvision.datasets.FashionMNIST(
    root = path_name + '/fashionmnist', download=True
)

print("MNIST and FashionMNIST datasets downloaded.")