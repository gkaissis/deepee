# %%
import torch
import pytorch_lightning as pl
import torchvision as tv
import segmentation_models_pytorch as smp

from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import albumentations as a
from datetime import datetime

from dataloader import (
    MSD_data_images,
    AlbumentationsTorchTransform,
    calc_mean_std,
    create_albu_transform,
)

from deepee import UniformDataLoader

# %%
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")


class args:
    batch_size = 32
    test_batch_size = 32
    lr = None  # set individually
    num_epochs = 20
    log_interval = 100
    img_size = 256
    data_dir = "data/MSD_Liver/"
    rotation = 45
    translate = 0.1
    scale = 0.1
    individual_albu_probs = 0.1
    noise_std = 0.05
    noise_prob = 0.25


# %%
basic_tfs = [
    a.Resize(args.img_size, args.img_size,),
    a.RandomCrop(args.img_size, args.img_size),
    a.ToFloat(max_value=255.0),
]
stats_tf_imgs = AlbumentationsTorchTransform(a.Compose(basic_tfs))
# dataset to calculate stats
trainset = MSD_data_images(args.data_dir + "/train", transform=stats_tf_imgs,)
# get stats
mean, std = calc_mean_std(trainset)

# change transforms based on stats
train_tf = create_albu_transform(args, mean, std)
trainset.transform = train_tf
# L_train = round(0.85 * len(trainset))
# trainset, valset = torch.utils.data.random_split(
#     trainset,
#     (L_train, len(trainset) - L_train),
#     generator=torch.Generator().manual_seed(42),
# )

# mask is a special keyword in albumentations
test_trans = a.Compose(
    [
        *basic_tfs,
        a.Normalize(mean, std, max_pixel_value=1.0),
        a.Lambda(
            image=lambda x, **kwargs: x.reshape(
                # add extra channel to be compatible with nn.Conv2D
                -1,
                args.img_size,
                args.img_size,
            ),
            mask=lambda x, **kwargs: np.where(
                # binarize masks
                x.reshape(-1, args.img_size, args.img_size) / 255.0 > 0.5,
                np.ones_like(x),
                np.zeros_like(x),
            ).astype(np.float32),
        ),
    ]
)

valset = MSD_data_images(
    args.data_dir + "/val", transform=AlbumentationsTorchTransform(test_trans),
)
testset = MSD_data_images(
    args.data_dir + "/test", transform=AlbumentationsTorchTransform(test_trans),
)
trainloader = UniformDataLoader(
    trainset,
    batch_size=args.batch_size,
    pin_memory=torch.cuda.is_available(),
    num_workers=0 if torch.cuda.is_available() else 32,
)
valloader = torch.utils.data.DataLoader(
    valset,
    batch_size=args.test_batch_size,
    pin_memory=torch.cuda.is_available(),
    num_workers=0 if torch.cuda.is_available() else 32,
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.test_batch_size,
    pin_memory=torch.cuda.is_available(),
    num_workers=0 if torch.cuda.is_available() else 32,
    shuffle=False,
)
# %%
seg_colormap = torch.tensor([[0, 0, 0], [255, 0, 0]])


def plot_torch_imgs(grid_imgs):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_imgs.permute(1, 2, 0))
    plt.axis("off")
    plt.show()


normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def visualize_seg(imgs, segs, gamma=0.5):
    imgs = normalize(imgs)
    if segs.sum().item() > 0:
        segs = normalize(segs)
    out = gamma * imgs + (1 - gamma) * segs
    plot_torch_imgs(make_grid(out))


def segmentation_to_RGB(segmentation):
    out = torch.zeros(
        (segmentation.shape[0], 3, segmentation.shape[2], segmentation.shape[3])
    )
    # now this is ugly but not worth to optimise
    for i in range(segmentation.shape[0]):
        for j in range(segmentation.shape[2]):
            for k in range(segmentation.shape[3]):
                out[i, :, j, k] = seg_colormap[segmentation[i, 0, j, k]]
    out = out.to(torch.float) / 255.0
    return out


# %%
img_segs = [trainset[i] for i in range(0, len(trainset), 200)]
img_batch = torch.stack([i[0] for i in img_segs])
seg_batch = torch.stack(
    [torch.from_numpy(np.asarray(i[1], dtype=np.int32)) for i in img_segs]
)
img_batch = normalize(img_batch)
segmentation_mask = segmentation_to_RGB(seg_batch)

# plot_batch = torch.vstack([img_batch, segmentation_mask])
# grid_img = make_grid(plot_batch, nrow=8)
# %%
visualize_seg(img_batch, segmentation_mask)
# %%

from deepee import ModelSurgeon, SurgicalProcedures

surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)
# %%
class SegmentationNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="vgg11_bn",
            encoder_weights="imagenet",
            classes=1,
            in_channels=1,
            activation="sigmoid",
        )
        surgeon.operate(self.model)
        self.loss_fn = smp.utils.losses.DiceLoss()

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
        score = smp.utils.functional.f_score(pred, target, threshold=0.5)
        out = {"val_loss": loss, "val_f_score": score}
        for key, value in out.items():
            self.log(key, value)
        return out

    def test_step(self, batch, batch_idx):
        data, target = batch
        pred = self.model(data)
        loss = self.loss_fn(pred, target)
        score = smp.utils.functional.f_score(pred, target, threshold=0.5)
        out = {"test_loss": loss, "test_f_score": score}
        for key, value in out.items():
            self.log(key, value)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def test(self, testloader):
        self.model.eval()
        preds, targets = [], []
        for data, target in testloader:
            preds.extend(self.model(data.to(self.device)).detach().cpu().tolist())
            targets.extend(target.tolist())
        score = smp.utils.functional.f_score(
            torch.tensor(preds), torch.tensor(targets), threshold=0.5
        ).item()
        out = {"f_score": score}
        return out


# %%
# model = SegmentationNetwork()
# temp_tr = pl.Trainer()
# lr_finder = temp_tr.tuner.lr_find(
#     model, train_dataloader=trainloader, min_lr=1e-5, max_lr=1e-1, num_training=50
# )
# # %%
# fig = lr_finder.plot(suggest=True)
# # plt.xlim(right=0.1)
# # plt.ylim(bottom=0.0, top=0.7)
# fig.show()
# %%
args.lr = 1e-2
args.num_epochs = 20
# %%
# results = []
# for i in range(5):
#     model = SegmentationNetwork()
#     logger = pl.loggers.TensorBoardLogger(
#         "logs", name=f"liver_segmentation_{timestamp}_v{i}"
#     )
#     trainer = pl.Trainer(
#         max_epochs=args.num_epochs,
#         gpus=1 if torch.cuda.is_available() else 0,
#         logger=logger,
#     )
#     trainer.fit(model, trainloader, valloader)
#     res = model.test(testloader)
#     print(res)
#     results.append(res)
# torch.save(results, f"vanilla_segmentation_{timestamp}.pt")
# # %%
# img_batch = torch.stack([testset[i][0] for i in range(0, len(testset), 25)])
# model.eval()
# seg_batch = model(img_batch.to(model.device)).detach().cpu()
# seg_batch = torch.where(seg_batch > 0.5, 1, 0)
# # %%
# visualize_seg(img_batch, segmentation_to_RGB(seg_batch))
# %%
from deepee.watchdog import PrivacyWatchdog, PrivacyBudgetExhausted
from deepee import PrivacyWrapper

# %%

#%
noise_mult = 5.0  # 3.0
clip_norm = 0.5  # 0.8
# %%
class PrivateSegmentationNetwork(SegmentationNetwork):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="vgg11_bn",
            encoder_weights="imagenet",
            classes=1,
            in_channels=1,
            activation="sigmoid",
        )
        self.loss_fn = smp.utils.losses.DiceLoss()
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
            L2_clip=clip_norm,
            noise_multiplier=noise_mult,
            watchdog=watchdog,
        )

    def on_after_backward(self):
        self.model.clip_and_accumulate()
        self.model.noise_gradient()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.model.prepare_next_batch()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


# %%
# model = PrivateSegmentationNetwork()
# temp_tr = pl.Trainer()
# lr_finder = temp_tr.tuner.lr_find(
#     model, train_dataloader=trainloader, min_lr=1e-5, max_lr=1e-1, num_training=50
# )
# # %%
# fig = lr_finder.plot(suggest=True)
# # plt.xlim(right=0.1)
# # plt.ylim(bottom=0.0, top=0.7)
# fig.show()
# %%
args.lr = 1e-2
args.num_epochs = 20
# %%
results = []
for i in range(5):
    logger = pl.loggers.TensorBoardLogger(
        "logs", name=f"liver_segmentation_DP_{timestamp}_v{i}"
    )
    private_model = PrivateSegmentationNetwork()
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=logger,
    )
    try:
        trainer.fit(private_model, trainloader, valloader)
    except PrivacyBudgetExhausted as e:
        print(f"Privacy budget is exhausted")
    # %%
    final_eps = private_model.model.current_epsilon
    print(f"Final epsilon: {final_eps:.2f}")
    # %%
    res = private_model.test(testloader)
    res["final_epsilon"] = final_eps
    print(res)
    results.append(res)
torch.save(results, f"DP_segmentation_results_{timestamp}.pt")
# %%
img_batch = torch.stack([testset[i][0] for i in range(0, len(testset), 25)])
seg_batch = (
    private_model.model.wrapped_model(img_batch.to(private_model.device)).detach().cpu()
)
seg_batch = torch.where(seg_batch > 0.5, 1, 0)
# %%
visualize_seg(img_batch, segmentation_to_RGB(seg_batch))
# %%
# rdp version
from deepee.watchdog import compute_rdp, rdp_privacy_spent

q = args.batch_size / len(trainset)
orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
steps_taken = args.num_epochs * len(trainloader)
rdp = compute_rdp(q, noise_mult, steps_taken, orders)
spent, _ = rdp_privacy_spent(orders=orders, rdp=rdp, delta=1e-5)
print(f"RDP epsilon: {spent}")
# %%
