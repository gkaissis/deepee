# Examples

This page shows the basic utilisation of `deepee` in a step by step MNIST example. 

Being by importing the relevant libraries:

```py
from deepee import (PrivacyWrapper, PrivacyWatchdog, UniformDataLoader,
                     ModelSurgeon, SurgicalProcedures)
import torch
from torch import nn
from torchvision import datasets, transforms

class args:
    batch_size = 200
    test_batch_size = 200
    log_interval = 1000
    num_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = args.device
```

To train with DP guarantees, a special DataLoader is required. `deepee` provides this DataLoader with sensible presets:

```py
train_loader = UniformDataLoader(
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
```

Next, define the network architecture:

```py
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        x = self.bn1(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
```

To train with DP, we now attach the `PrivacyWrapper` to the model and set up a `PrivacyWatchDog` to monitor the privacy loss:

```py
watchdog = PrivacyWatchdog(
    train_loader,
    target_epsilon=1.0,
    abort=False,
    target_delta=1e-5,
    fallback_to_rdp=False,
)

model = PrivacyWrapper(SimpleNet(), args.batch_size, 1.0, 1.0, watchdog=watchdog).to(
    args.device
)
optimizer = torch.optim.SGD(model.wrapped_model.parameters(), lr=0.1)
```

The `PrivacyWrapper` will throw an error now:

```py
---------------------------------------------------------------------------
BadModuleError                            Traceback (most recent call last)
<ipython-input-16-a70572be6dfb> in <module>()
      7 )
      8 
----> 9 model = PrivacyWrapper(SimpleNet(), args.batch_size, 1.0, 1.0, watchdog=watchdog).to(
     10     args.device
     11 )

1 frames
/usr/local/lib/python3.7/dist-packages/deepee/snooper.py in snoop(self, model)
     38             msg += validator.validate(model)
     39         if msg != "":
---> 40             raise BadModuleError(msg)
     41 
     42 

BadModuleError: BatchNorm Layers must have track_running_stats turned off, otherwise be replaced with InstanceNorm, LayerNorm or GroupNorm.
```

Luckily, this modification can be easily done using the `ModelSurgeon`:

```py
surgeon = ModelSurgeon(SurgicalProcedures.BN_to_GN)
model = surgeon.operate(model) 
```

We can now proceed with training as usual:

```py
# Train
for epoch in range(args.num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
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
```

```
Train Epoch: 0 [0/60000 (0%)]	Loss: 2.317095
INFO:root:Privacy spent at 200 steps: 0.27
INFO:root:Privacy spent at 300 steps: 0.34
INFO:root:Privacy spent at 400 steps: 0.39

Test set: Average loss: 1.6950, Accuracy: 5639/10000 (56%)
Train Epoch: 1 [0/60000 (0%)]	Loss: 1.667842
INFO:root:Privacy spent at 500 steps: 0.44
INFO:root:Privacy spent at 600 steps: 0.49
```
