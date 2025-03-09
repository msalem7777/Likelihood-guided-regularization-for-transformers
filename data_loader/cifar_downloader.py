import torchvision

path_name = "."

# Download CIFAR-10
torchvision.datasets.CIFAR10(
    root = path_name + '/cifar10', download=True
)

# Download CIFAR-100
cifar100_train = torchvision.datasets.CIFAR100(
    root = path_name + '/cifar100', download=True
)

print("CIFAR-10 and CIFAR-100 datasets downloaded.")