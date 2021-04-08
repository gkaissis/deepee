# %%
import torch
import pytorch_lightning as pl
import torchvision as tv

from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from sklearn import metrics

from PIL import Image
from collections import Counter
from tqdm import tqdm

from deepee import UniformDataLoader

# %%
class args:
    batch_size = 32
    test_batch_size = 200
    lr = 1e-3
    num_epochs = 5
    log_interval = 100


# %%
def single_channel_loader(filename):
    """Converts `filename` to a grayscale PIL Image"""
    with open(filename, "rb") as f:
        img = Image.open(f).convert("L")
        return img.copy()


trainset = tv.datasets.ImageFolder(
    "./data/pneumonia/train/",
    transform=tv.transforms.Compose(
        [
            tv.transforms.RandomAffine(
                degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            tv.transforms.Resize(224),
            tv.transforms.RandomCrop((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4814,), (0.2377,)),
            tv.transforms.Lambda(
                lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                    x, 3, dim=0
                )
            ),
        ]
    ),
    # target_transform=tv.transforms.ToTensor(),
    loader=single_channel_loader,
)

L_train = round(0.85 * len(trainset))
trainset, valset = torch.utils.data.random_split(
    trainset,
    (L_train, len(trainset) - L_train),
    generator=torch.Generator().manual_seed(42),
)
testset = tv.datasets.ImageFolder(
    "./data/pneumonia/test/",
    transform=tv.transforms.Compose(
        [
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4814,), (0.2377,)),
            tv.transforms.Lambda(
                lambda x: torch.repeat_interleave(  # pylint: disable=no-member
                    x, 3, dim=0
                )
            ),
        ]
    ),
    # target_transform=tv.transforms.ToTensor(),
    loader=single_channel_loader,
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
    shuffle=True,
)
# %%
img_batch = torch.stack([trainset[i][0] for i in range(32)])
grid_img = make_grid(img_batch, nrow=8)
grid_img = (grid_img - torch.min(grid_img)) / (
    torch.max(grid_img) - torch.min(grid_img)
)
plt.figure(figsize=(10, 5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.show()
# %%
targets = []
for _, target in tqdm(trainloader, total=len(trainloader), leave=False):
    targets.extend(target.tolist())
# %%
target_distribution = Counter(targets)
class_weights = torch.tensor(
    [
        1.0 - (target_distribution[key] / len(trainset))
        for key in sorted(target_distribution)
    ]
) * len(target_distribution)

# %%
from deepee import ModelSurgeon, SurgicalProcedures

surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)
# %%
logger = pl.loggers.TensorBoardLogger("logs", name="pneumonia_classification")

# %%
class PLClassifier(pl.LightningModule):
    def __init__(self, class_weights):
        super().__init__()
        self.classifier = tv.models.resnet18(pretrained=True)
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.classifier.fc = torch.nn.Linear(512, 1)
        for param in self.classifier.fc.parameters():
            param.requires_grad = True
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        self.weight = class_weights
        surgeon.operate(self.classifier)  # for comparison reasons

    def forward(self, x):
        embedding = self.classifier(x)
        return embedding

    def training_step(self, batch, batch_idx):
        data, target = batch
        pred = self.classifier(data).squeeze()
        loss = self.loss_fn(
            pred,
            target.to(torch.float),
            weight=self.weight[target.data.view(-1).long()]
            .view_as(target)
            .to(self.device),
        )
        self.log("train_loss", loss)
        output = {"loss": loss}
        return output

    def validation_step(self, batch, batch_idx):
        data, target = batch
        pred = self.classifier(data).squeeze()
        loss = self.loss_fn(pred, target.to(torch.float))
        pred = pred.detach().cpu()
        pred_classes = torch.where(pred < 0.5, 0, 1).tolist()
        target = target.cpu().detach().tolist()
        mcc = metrics.matthews_corrcoef(target, pred_classes)
        acc = metrics.accuracy_score(target, pred_classes)
        roc = metrics.roc_auc_score(target, pred)
        out = {"val_loss": loss, "val_mcc": mcc, "val_accuracy": acc, "val_roc": roc}
        for key, value in out.items():
            self.log(key, value)
        return out

    def test_step(self, batch, batch_idx):
        data, target = batch
        target = target
        pred = self.classifier(data).squeeze()
        loss = self.loss_fn(pred, target.to(torch.float))
        pred = pred.detach().cpu()
        pred_classes = torch.where(pred < 0.5, 0, 1).tolist()
        target = target.cpu().detach().tolist()
        mcc = metrics.matthews_corrcoef(target, pred_classes)
        acc = metrics.accuracy_score(target, pred_classes)
        roc = metrics.roc_auc_score(target, pred)
        out = {
            "test_loss": loss,
            "test_mcc": mcc,
            "test_accuracy": acc,
            "test_roc": roc,
        }
        for key, value in out.items():
            self.log(key, value)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    def on_epoch_start(self):
        if self.current_epoch == 1:
            print("Unfreeze normalization layers")
            for layer_name, layer in self.classifier.wrapped_model.named_modules():
                if (
                    isinstance(layer, torch.nn.BatchNorm2d)
                    or isinstance(layer, torch.nn.GroupNorm)
                    or isinstance(layer, torch.nn.Linear)
                ):
                    for p in layer.parameters():
                        p.requires_grad = True
        if self.current_epoch == 2:
            print("Unfreeze model")
            for param in self.classifier.parameters():
                param.requires_grad = True


# %%
classifier = PLClassifier(class_weights)
trainer = pl.Trainer(
    max_epochs=args.num_epochs,
    gpus=1 if torch.cuda.is_available() else 0,
    logger=logger,
)
# %%
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
    def __init__(self, class_weights):
        super().__init__()
        model = tv.models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 1)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(512, 1)
        for param in model.fc.parameters():
            param.requires_grad = True
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        self.weight = class_weights
        surgeon.operate(model)
        watchdog = PrivacyWatchdog(
            trainloader,
            target_epsilon=10.0,
            abort=True,
            target_delta=1e-5,
            report_every_n_steps=len(trainloader),
            # fallback_to_rdp=True,
        )
        self.classifier = PrivacyWrapper(
            model,
            args.batch_size,
            0.3,
            0.5,
            watchdog=watchdog,  # watchdog,
        )

    def forward(self, x):
        embedding = self.classifier(x)
        return embedding

    def training_step(self, batch, batch_idx):
        data, target = batch
        pred = self.classifier(data).squeeze()
        loss = self.loss_fn(
            pred,
            target.to(torch.float),
            weight=self.weight[target.data.view(-1).long()]
            .view_as(target)
            .to(self.device),
        )
        self.log("train_loss", loss)
        output = {"loss": loss}
        return output

    def validation_step(self, batch, batch_idx):
        data, target = batch
        pred = self.classifier(data).squeeze()
        loss = self.loss_fn(pred, target.to(torch.float))
        pred_classes = torch.where(pred.detach().cpu() < 0.5, 0, 1).tolist()
        target = target.cpu().detach().tolist()
        mcc = metrics.matthews_corrcoef(target, pred_classes)
        acc = metrics.accuracy_score(target, pred_classes)
        out = {"val_loss": loss, "val_mcc": mcc, "val_accuracy": acc}
        for key, value in out.items():
            self.log(key, value)
        return out

    def test_step(self, batch, batch_idx):
        data, target = batch
        target = target
        pred = self.classifier(data).squeeze()
        loss = self.loss_fn(pred, target.to(torch.float))
        pred_classes = torch.where(pred.detach().cpu() < 0.5, 0, 1).tolist()
        target = target.cpu().detach().tolist()
        mcc = metrics.matthews_corrcoef(target, pred_classes)
        acc = metrics.accuracy_score(target, pred_classes)
        out = {"test_loss": loss, "test_mcc": mcc, "test_accuracy": acc}
        for key, value in out.items():
            self.log(key, value)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    def on_after_backward(self):
        self.classifier.clip_and_accumulate()
        self.classifier.noise_gradient()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.classifier.prepare_next_batch()

    def on_epoch_start(self):
        if self.current_epoch == 1:
            print("Unfreeze normalization layers")
            for layer_name, layer in self.classifier.wrapped_model.named_modules():
                if (
                    isinstance(layer, torch.nn.BatchNorm2d)
                    or isinstance(layer, torch.nn.GroupNorm)
                    or isinstance(layer, torch.nn.Linear)
                ):
                    for p in layer.parameters():
                        p.requires_grad = True
            self.classifier.update_clones()
        if self.current_epoch == 2:
            print("Unfreeze model")
            for param in self.classifier.wrapped_model.parameters():
                param.requires_grad = True
            self.classifier.update_clones()


# %%
privateclassifier = PrivatePLClassifier(class_weights)
privatetrainer = pl.Trainer(
    max_epochs=args.num_epochs,
    weights_summary=None,
    gpus=1 if torch.cuda.is_available() else 0,
    logger=logger,
)
# %%
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
