from torch import optim
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules
import torch
import torchvision
from memory_profiler import profile
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="vgg11_bn",
    encoder_weights="imagenet",
    classes=1,
    in_channels=1,
    activation="sigmoid",
)

model = convert_batchnorm_modules(model)

dataset = torchvision.datasets.FakeData(
    size=32,
    image_size=(1, 256, 256),
    num_classes=2,
    transform=torchvision.transforms.ToTensor(),
)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, num_workers=0, pin_memory=False
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
privacy_engine = PrivacyEngine(
    model,
    sample_rate=1.0,
    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    secure_rng=False,
)

criterion = smp.utils.losses.DiceLoss()


@profile
def main():
    model.train()
    optimizer.zero_grad()
    data, _ = next(iter(dataloader))
    output = model(data)
    loss = criterion(data, output)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
