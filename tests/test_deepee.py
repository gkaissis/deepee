from deepee import __version__, PrivacyWrapper, UniformDataLoader
import torch
import toml
from pathlib import Path


def test_version():
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = toml.loads(open(str(path)).read())
    assert __version__ == pyproject["tool"]["poetry"]["version"]


def test_overfitting():
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

    assert loss.item() < 0.01