from torch import optim
from deepee import PrivacyWrapper, UniformDataLoader, ModelSurgeon, SurgicalProcedures
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


surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)
model = surgeon.operate(model)
model = PrivacyWrapper(
    model, num_replicas=32, L2_clip=1.0, noise_multiplier=1.0, secure_rng=False
)

dataset = torchvision.datasets.FakeData(
    size=32,
    image_size=(1, 256, 256),
    num_classes=2,
    transform=torchvision.transforms.ToTensor(),
)

dataloader = UniformDataLoader(dataset, batch_size=32, num_workers=0, pin_memory=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

criterion = smp.utils.losses.DiceLoss()


@profile
def main():
    model.train()
    optimizer.zero_grad()
    data, _ = next(iter(dataloader))
    output = model(data)
    loss = criterion(data, output)
    loss.backward()
    model.clip_and_accumulate()
    model.noise_gradient()
    model.prepare_next_batch()
    optimizer.step()


if __name__ == "__main__":
    main()
