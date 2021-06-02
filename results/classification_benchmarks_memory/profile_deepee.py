import os

os.environ["OMP_NUM_THREADS"] = "16"
import torch

torch.set_num_threads(16)
torch.set_num_interop_threads(16)
from deepee import PrivacyWrapper, UniformDataLoader, ModelSurgeon, SurgicalProcedures
import torchvision
from memory_profiler import profile
import gc

gc.disable()

model = torchvision.models.vgg11_bn()
model.classifier = torch.nn.Linear(512 * 7 * 7, 1)


surgeon = ModelSurgeon(SurgicalProcedures.BN_to_GN)
model = surgeon.operate(model)
model = PrivacyWrapper(
    model, num_replicas=32, L2_clip=1.0, noise_multiplier=1.0, secure_rng=False
)

dataset = torchvision.datasets.FakeData(
    size=32,
    image_size=(3, 256, 256),
    num_classes=2,
    transform=torchvision.transforms.ToTensor(),
)

dataloader = UniformDataLoader(dataset, batch_size=32, num_workers=0, pin_memory=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

criterion = torch.nn.BCEWithLogitsLoss()


@profile
def main():
    model.train()
    optimizer.zero_grad()
    data, label = next(iter(dataloader))
    output = model(data)
    loss = criterion(label[..., None].to(float), output)
    loss.backward()
    model.clip_and_accumulate()
    model.noise_gradient()
    model.prepare_next_batch()
    optimizer.step()


if __name__ == "__main__":
    print("Deepee")
    main()
