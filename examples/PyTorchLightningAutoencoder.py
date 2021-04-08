# %% [markdown]
# # Deepee with Pytorch Lightning

# %%
import torch
import pytorch_lightning as pl
import torchvision as tv

from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from sklearn import metrics

from deepee import UniformDataLoader

# %%
class args:
    batch_size = 32
    test_batch_size = 200
    lr = 1e-3
    num_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval = 100


# %%
trainset = tv.datasets.MNIST(
    "./data/",
    train=True,
    download=True,
    transform=tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

L_train = round(0.85 * len(trainset))
trainset, valset = torch.utils.data.random_split(
    trainset,
    (L_train, len(trainset) - L_train),
    generator=torch.Generator().manual_seed(42),
)
testset = tv.datasets.MNIST(
    "./data/",
    train=False,
    download=True,
    transform=tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)
trainloader = UniformDataLoader(
    trainset,
    batch_size=args.batch_size,
    pin_memory=torch.cuda.is_available(),
    num_workers=0 if torch.cuda.is_available() else 32,
)
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.batch_size,
    pin_memory=torch.cuda.is_available(),
    num_workers=0 if torch.cuda.is_available() else 32,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch_size,
    pin_memory=torch.cuda.is_available(),
    num_workers=0 if torch.cuda.is_available() else 32,
    shuffle=False,
)
# %%
class LinearDecoder(torch.nn.Module):
    def __init__(self):
        super(LinearDecoder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 128), torch.nn.ReLU(), torch.nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        return self.model(x)


class LinearEncoder(torch.nn.Module):
    def __init__(self):
        super(LinearEncoder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


class LinearAutoencoder(torch.nn.Module):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        embd = self.encoder(x)
        return self.decoder(embd)


class ConvAutoencoder(torch.nn.Module):
    def __init__(self, unet=False):
        super(ConvAutoencoder, self).__init__()
        self.unet = unet
        ## encoder
        self.e1 = torch.nn.Conv2d(1, 8, 3)
        self.e2 = torch.nn.ELU()
        self.e3 = torch.nn.MaxPool2d(2, return_indices=True)
        self.e4 = torch.nn.Conv2d(8, 32, 3)
        self.e5 = torch.nn.ELU()
        self.e6 = torch.nn.MaxPool2d(2, return_indices=True)
        self.e7 = torch.nn.Conv2d(32, 64, 3)
        self.e8 = torch.nn.ELU()
        self.e9 = torch.nn.Conv2d(64, 128, 3)
        ## decoder
        self.d9 = torch.nn.ConvTranspose2d(128, 64, 3)
        self.d8 = torch.nn.ELU()
        self.d7 = torch.nn.ConvTranspose2d(64, 32, 3)
        self.d6 = torch.nn.MaxUnpool2d(2)
        self.d5 = torch.nn.ELU()
        self.d4 = torch.nn.ConvTranspose2d(32, 8, 3)
        self.d3 = torch.nn.MaxUnpool2d(2)
        self.d2 = torch.nn.ELU()
        self.d1 = torch.nn.ConvTranspose2d(8, 1, 3)
        self.final = torch.nn.Tanh()

    def encode(self, x, return_intermediates=False):
        # encode
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3[0])
        x5 = self.e5(x4)
        x6 = self.e6(x5)
        x7 = self.e7(x6[0])
        x8 = self.e8(x7)
        x9 = self.e9(x8)
        if return_intermediates:
            return [x1, x2, x3, x4, x5, x6, x7, x8, x9]
        else:
            return x9

    def decode(self, intermediates):
        x8 = self.d9(intermediates[8])
        if self.unet:
            x8 += intermediates[7]
        x7 = self.d8(x8)
        x6 = self.d7(x7)
        if self.unet:
            x6 += intermediates[5][0]
        x5 = self.d6(x6, intermediates[5][1], output_size=intermediates[4].shape)
        x4 = self.d5(x5)
        x3 = self.d4(x4)
        if self.unet:
            x3 += intermediates[2][0]
        x2 = self.d3(x3, intermediates[2][1], output_size=intermediates[1].shape)
        x1 = self.d2(x2)
        out = self.d1(x1)
        out = self.final(out) * 5.0
        return out

    def forward(self, x):
        intm = self.encode(x, return_intermediates=True)
        out = self.decode(intm)
        return out


# %%
class PLAutoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvAutoencoder()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        self.log("train_loss", loss)
        output = {"loss": loss}
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        self.log("val_loss", loss)
        out = {"val_loss": loss}
        return out

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        self.log("test_loss", loss)
        out = {"test_loss": loss}
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer


# %%
autoencoder = PLAutoencoder()
trainer = pl.Trainer(
    max_epochs=args.num_epochs, gpus=1 if torch.cuda.is_available() else 0
)
trainer.fit(autoencoder, trainloader, valloader)

# %%
trainer.test(autoencoder, testloader)

# %%
in_batch = torch.stack([testset[i][0] for i in range(32)])
img_batch = autoencoder(
    in_batch.to("cuda" if torch.cuda.is_available() else "cpu")
).cpu()
grid_img = make_grid(img_batch, nrow=8)
grid_img = (grid_img - torch.min(grid_img)) / (
    torch.max(grid_img) - torch.min(grid_img)
)
plt.figure(figsize=(10, 5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.show()
# %%
from deepee.watchdog import PrivacyWatchdog
from deepee import PrivacyWrapper

# %%
# now privately


class PrivatePLAutoencoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvAutoencoder()
        watchdog = PrivacyWatchdog(
            trainloader,
            target_epsilon=10.0,
            abort=True,
            target_delta=1e-5,
            report_every_n_steps=len(trainloader),
        )
        self.model = PrivacyWrapper(
            self.model,
            args.batch_size,
            1.0,
            0.5,
            watchdog=watchdog,
        ).to(args.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        self.log("train_loss", loss)
        output = {"loss": loss}
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        self.log("val_loss", loss)
        out = {"val_loss": loss}
        return out

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        self.log("test_loss", loss)
        out = {"test_loss": loss}
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    def on_after_backward(self):
        self.model.clip_and_accumulate()
        self.model.noise_gradient()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.model.prepare_next_batch()


# %%
privateAE = PrivatePLAutoencoder()
privatetrainer = pl.Trainer(
    max_epochs=args.num_epochs,
    weights_summary=None,
    gpus=1 if torch.cuda.is_available() else 0,
)
privatetrainer.fit(privateAE, train_dataloader=trainloader, val_dataloaders=valloader)
# %%
print(f"Final epsilon: {privateAE.model.current_epsilon:.2f}")

# %%
in_batch = torch.stack([testset[i][0] for i in range(32)])
img_batch = privateAE(in_batch)
grid_img = make_grid(img_batch, nrow=8)
grid_img = (grid_img - torch.min(grid_img)) / (
    torch.max(grid_img) - torch.min(grid_img)
)
plt.figure(figsize=(10, 5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.show()
# %%
