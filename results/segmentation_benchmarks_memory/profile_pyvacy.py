import os

os.environ["OMP_NUM_THREADS"] = "16"
import torch

torch.set_num_threads(16)
torch.set_num_interop_threads(16)
from pyvacy.optim import DPSGD
from pyvacy.sampling import get_data_loaders
from opacus.utils.module_modification import convert_batchnorm_modules
import torchvision
from memory_profiler import profile
import segmentation_models_pytorch as smp
import gc

gc.disable()

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

minibatch_loader, microbatch_loader = get_data_loaders(32, 1, 1)
dataloader = minibatch_loader(dataset)

optimizer = DPSGD(
    params=model.parameters(),
    lr=1.0,
    l2_norm_clip=1.0,
    noise_multiplier=1.0,
    minibatch_size=32,
    microbatch_size=1,
)

criterion = smp.utils.losses.DiceLoss()


@profile
def main():
    model.train()
    optimizer.zero_grad()
    X_minibatch, y_minibatch = next(iter(dataloader))
    for X_microbatch, _ in microbatch_loader(
        torch.utils.data.TensorDataset(X_minibatch, y_minibatch)
    ):
        optimizer.zero_microbatch_grad()
        output = model(X_microbatch)
        loss = criterion(X_microbatch, output)
        loss.backward()
        optimizer.microbatch_step()
    optimizer.step()


if __name__ == "__main__":
    print("Pyvacy")
    main()
