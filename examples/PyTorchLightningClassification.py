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
class PLClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        embedding = self.classifier(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        self.log("train_loss", loss)
        output = {"loss": loss}
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        self.log("val_loss", loss)
        pred_classes = pred.detach().cpu().argmax(dim=1)
        mcc = metrics.matthews_corrcoef(y.cpu().detach().tolist(), pred_classes)
        acc = metrics.accuracy_score(y.cpu().detach().tolist(), pred_classes)
        out = {"val_loss": loss, "val_mcc": mcc, "val_accuracy": acc}
        return out

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        self.log("test_loss", loss)
        pred_classes = pred.detach().cpu().argmax(dim=1)
        mcc = metrics.matthews_corrcoef(y.cpu().detach().tolist(), pred_classes)
        acc = metrics.accuracy_score(y.cpu().detach().tolist(), pred_classes)
        out = {"test_loss": loss, "test_mcc": mcc, "test_accuracy": acc}
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer


# %%
classifier = PLClassifier()
trainer = pl.Trainer(
    max_epochs=args.num_epochs, gpus=1 if torch.cuda.is_available() else 0
)
trainer.fit(classifier, trainloader, valloader)

# %%
trainer.test(classifier, trainloader)
trainer.test(classifier, testloader)

# %%
from deepee.watchdog import PrivacyWatchdog, PrivacyBudgetExhausted
from deepee import PrivacyWrapper


# %%
# now privately


class PrivatePLClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
        )
        watchdog = PrivacyWatchdog(
            trainloader,
            target_epsilon=10.0,
            abort=True,
            target_delta=1e-5,
            report_every_n_steps=len(trainloader),
        )
        self.classifier = PrivacyWrapper(
            model,
            args.batch_size,
            1.0,
            0.5,
            watchdog=watchdog,  # watchdog,
        ).to(args.device)

    def forward(self, x):
        embedding = self.classifier.wrapped_model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        self.log("train_loss", loss)
        output = {"loss": loss}
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        pred_classes = pred.detach().cpu().argmax(dim=1)
        mcc = metrics.matthews_corrcoef(y.cpu().detach().tolist(), pred_classes)
        acc = metrics.accuracy_score(y.cpu().detach().tolist(), pred_classes)
        self.log("val_loss", loss)
        self.log("val_mcc", mcc)
        out = {"val_loss": loss, "val_mcc": mcc, "val_accuracy": acc}
        return out

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        pred = self.classifier(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        pred_classes = pred.detach().cpu().argmax(dim=1)
        mcc = metrics.matthews_corrcoef(y.cpu().detach().tolist(), pred_classes)
        acc = metrics.accuracy_score(y.cpu().detach().tolist(), pred_classes)
        self.log("test_loss", loss)
        self.log("test_mcc", mcc)
        out = {"test_loss": loss, "test_mcc": mcc, "test_accuracy": acc}
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.classifier.wrapped_model.parameters(), lr=args.lr
        )
        return optimizer

    def on_after_backward(self):
        self.classifier.clip_and_accumulate()
        self.classifier.noise_gradient()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.classifier.prepare_next_batch()


# %%
privateclassifier = PrivatePLClassifier()
privatetrainer = pl.Trainer(
    max_epochs=args.num_epochs,
    weights_summary=None,
    gpus=1 if torch.cuda.is_available() else 0,
)
try:
    privatetrainer.fit(privateclassifier, trainloader, valloader)
except PrivacyBudgetExhausted as e:
    print(f"Privacy budget is exhausted")

# %%
print(f"Final epsilon: {privateclassifier.classifier.current_epsilon:.2f}")
# %%
privatetrainer.test(privateclassifier, trainloader)
privatetrainer.test(privateclassifier, testloader)

# %%
