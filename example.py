from deepee import PrivacyWrapper
import torch
from torch import nn
from torchvision import datasets, transforms

# from tqdm import tqdm
from deepee import UniformDataLoader


class args:
    batch_size = 64
    test_batch_size = 64
    log_interval = 1000
    num_epochs = 5


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.lin(x)


class DS(torch.utils.data.Dataset):
    def __init__(self):
        self.features = torch.linspace(0, 1, 1000).requires_grad_(True)
        self.labels = torch.linspace(0, 1, 1000).requires_grad_(True)

    def __getitem__(self, idx):
        return (self.features, self.labels)

    def __len__(self):
        return len(self.features)


dl = UniformDataLoader(DS(), batch_size=2)

model = PrivacyWrapper(Model, 2, 1.0, 1.0)

optimizer = torch.optim.Adam(model.wrapped_model.parameters(), lr=4e-3)

for feature, label in dl:
    output = model(feature[..., None])
    loss = ((output - label[..., None]) ** 2).mean()
    loss.backward()
    model.clip_and_accumulate()
    model.noise_gradient()
    optimizer.step()
    model.prepare_next_batch()
    print(loss.item())

exit()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=args.test_batch_size,
    shuffle=True,
)


# class SimpleNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.fc2 = nn.Linear(256, 64)
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = torch.sigmoid(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         x = self.fc3(x)
#         return x


# class PretrainedNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.fc2 = nn.Linear(256, 64)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = torch.sigmoid(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x


# class FreshClassifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc3 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = self.fc3(x)
#         return x


# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.pretrained_part = PretrainedNet()
#         # for param in self.pretrained_part.parameters():
#         #     param.requires_grad_(False)
#         self.dp_part = PrivacyWrapper(
#             FreshClassifier,
#             num_replicas=args.batch_size,
#             L2_clip=100,
#             noise_multiplier=0,
#         )

#     def forward(self, x):
#         x = self.pretrained_part(x)
#         x = self.dp_part(x)
#         return x


# model = Model()

from torchvision.models import vgg16

model = PrivacyWrapper(vgg16, args.batch_size, L2_clip=100, noise_multiplier=0.0)

optimizer = torch.optim.SGD(model.wrapped_model.parameters(), lr=0.1)

device = "cpu"

data = torch.randn(64, 3, 224, 224)

output = model(data)

while True:
    pass

# Train
for epoch in range(args.num_epochs):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        model.clip_and_accumulate()
        model.noise_gradient()
        optimizer.step()
        model.prepare_next_batch()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    # Test
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.CrossEntropyLoss(reduction="sum")(
                output, target
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
