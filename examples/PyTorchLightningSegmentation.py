# %%
import torch
import pytorch_lightning as pl
import torchvision as tv
import segmentation_models_pytorch as smp

from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from sklearn import metrics
from numpy import asarray, int32

from deepee import UniformDataLoader

# %%
class args:
    batch_size = 32
    test_batch_size = 200
    lr = 1e-4
    num_epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval = 100


# %%
def to_one_hot(labelmask):
    h, w = labelmask.shape
    n_classes = 21 + 1  # 21 regular + 255 undefined
    one_hot = torch.zeros(n_classes, h, w)
    labelmask = torch.where(labelmask == 255, 21, labelmask)
    labelmask.unsqueeze_(0)
    one_hot.scatter_(0, labelmask, 1)
    return one_hot


# %%
trainset = tv.datasets.VOCSegmentation(
    "./data/",
    image_set="train",
    download=True,
    transform=tv.transforms.Compose(
        [
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    target_transform=tv.transforms.Compose(
        [
            tv.transforms.Resize(
                (224, 224), interpolation=tv.transforms.InterpolationMode.NEAREST
            ),
            tv.transforms.Lambda(
                lambda x: torch.from_numpy(asarray(x, dtype=int32)).type(
                    torch.LongTensor
                )
            ),
            tv.transforms.Lambda(lambda x: to_one_hot(x)),
        ]
    ),
)
L_train = round(0.85 * len(trainset))
trainset, valset = torch.utils.data.random_split(
    trainset,
    (L_train, len(trainset) - L_train),
    generator=torch.Generator().manual_seed(42),
)
testset = tv.datasets.VOCSegmentation(
    "./data/",
    image_set="val",
    download=True,
    transform=tv.transforms.Compose(
        [
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    target_transform=tv.transforms.Compose(
        [
            tv.transforms.Resize(
                (224, 224), interpolation=tv.transforms.InterpolationMode.NEAREST
            ),
            tv.transforms.Lambda(
                lambda x: torch.from_numpy(asarray(x, dtype=int32)).type(
                    torch.LongTensor
                )
            ),
            tv.transforms.Lambda(lambda x: to_one_hot(x)),
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
VOC_COLORMAP = torch.tensor(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
        [255, 255, 255],  # undefined
    ],
    dtype=torch.int,
)


def plot_torch_imgs(grid_imgs):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_imgs.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


def segmentation_to_RGB(segmentation):
    out = torch.zeros(
        (segmentation.shape[0], 3, segmentation.shape[2], segmentation.shape[3])
    )
    # now this is ugly but not worth to optimise
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[2]):
            for k in range(segmentation.shape[3]):
                out[i, :, j, k] = VOC_COLORMAP[segmentation[i, :, j, k].argmax()]
    out = out.to(torch.float) / 255.0
    return out


normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def visualize_seg(imgs, segs, gamma=0.5):
    imgs = normalize(imgs)
    if segs.sum().item() > 0:
        segs = normalize(segs)
    out = gamma * imgs + (1 - gamma) * segs
    plot_torch_imgs(make_grid(out))


# %%
img_batch = torch.stack([trainset[i][0] for i in range(32)])
seg_batch = torch.stack(
    [torch.from_numpy(asarray(trainset[i][1], dtype=int32)) for i in range(32)]
)
img_batch = normalize(img_batch)
segmentation_mask = segmentation_to_RGB(seg_batch)

# plot_batch = torch.vstack([img_batch, segmentation_mask])
# grid_img = make_grid(plot_batch, nrow=8)
# %%
visualize_seg(img_batch, segmentation_mask)
# %%
class SegmentationNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="vgg11_bn", encoder_weights="imagenet", classes=22
        )
        self.loss_fn = smp.utils.losses.DiceLoss(
            activation="sigmoid",
            ignore_channels=[22],
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        pred = self.model(data)
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss)
        output = {"loss": loss}
        return output

    def validation_step(self, batch, batch_idx):
        data, target = batch
        pred = self.model(data)
        loss = self.loss_fn(pred, target)
        self.log("val_loss", loss)
        out = {"val_loss": loss}
        return out

    def test_step(self, batch, batch_idx):
        data, target = batch
        pred = self.model(data)
        loss = self.loss_fn(pred, target)
        score = smp.utils.functional.f_score(pred, target, threshold=0.5)
        out = {"test_loss": loss, "f score": score}
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        return optimizer


# %%
model = SegmentationNetwork()
trainer = pl.Trainer(
    max_epochs=args.num_epochs, gpus=1 if torch.cuda.is_available() else 0
)
trainer.fit(model, trainloader, valloader)
# %%
trainer.test(model, testloader)
# %%
img_batch = torch.stack([testset[i][0] for i in range(32)])
model.eval()
seg_batch = model(img_batch.to(args.device)).detach().cpu()
# %%
visualize_seg(img_batch, segmentation_to_RGB(seg_batch))
# %%
from deepee.watchdog import PrivacyWatchdog, PrivacyBudgetExhausted
from deepee import PrivacyWrapper
from deepee import ModelSurgeon, SurgicalProcedures

surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)

# %%
class PrivateSegmentationNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="vgg11_bn", encoder_weights="imagenet", classes=22
        )

        self.loss_fn = smp.utils.losses.DiceLoss(
            activation="sigmoid",
            # ignore_channels=[21],
        )
        surgeon.operate(self.model)

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
            1.0,
            watchdog=watchdog,
        ).to(args.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        pred = self.model(data)
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss)
        output = {"loss": loss}
        return output

    def validation_step(self, batch, batch_idx):
        data, target = batch
        pred = self.model(data)
        loss = self.loss_fn(pred, target)
        self.log("val_loss", loss)
        out = {"val_loss": loss}
        return out

    def test_step(self, batch, batch_idx):
        data, target = batch
        pred = self.model(data)
        loss = self.loss_fn(pred, target)
        self.log("test_loss", loss)
        score = smp.utils.functional.f_score(pred, target, threshold=0.5)
        out = {"test_loss": loss, "f score": score}
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_after_backward(self):
        self.model.clip_and_accumulate()
        self.model.noise_gradient()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.model.prepare_next_batch()


# %%
private_model = PrivateSegmentationNetwork()
trainer = pl.Trainer(
    max_epochs=args.num_epochs,
    weights_summary=None,
    gpus=1 if torch.cuda.is_available() else 0,
)
try:
    trainer.fit(private_model, trainloader, valloader)
except PrivacyBudgetExhausted as e:
    print(f"Privacy budget is exhausted")
# %%
print(f"Final epsilon: {private_model.model.current_epsilon:.2f}")
# %%
trainer.test(private_model, testloader)
# %%
img_batch = torch.stack([testset[i][0] for i in range(32)])
seg_batch = private_model.model.wrapped_model(img_batch.to(args.device)).detach().cpu()
# %%
visualize_seg(img_batch, segmentation_to_RGB(seg_batch))
# %%