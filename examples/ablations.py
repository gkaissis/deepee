# %% [markdown]
# # Reviewer: more experiments are needed
# Like more datasets
# or impact of parameters
#
# ## Possible Datasets:
# - MNIST
# - CIFAR
# - MedNIST
#
# ## Parameters (models)
# - VGG11 - VGG19
# %%
import torch
import torchvision as tv
import numpy as np
from torchvision.transforms.transforms import Grayscale
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn import metrics

from deepee import UniformDataLoader
from deepee import ModelSurgeon, SurgicalProcedures

from deepee.watchdog import PrivacyWatchdog, PrivacyBudgetExhausted
from deepee import PrivacyWrapper

from opacus import PrivacyEngine
from opacus.utils import module_modification


# %%
class args:
    batch_size = 32
    test_batch_size = 200
    lr = 1e-3
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    overfit = None
    noise_mult = 0.5
    clip_norm = 0.25
    delta = 1e-5


trainname = f"cifar_{args.num_epochs}epochs_noise{args.noise_mult}_clip{args.clip_norm}"
if "cifar" in trainname:
    trainset = tv.datasets.CIFAR10(
        root="./data/",
        train=True,
        download=True,
        transform=tv.transforms.Compose(
            [
                tv.transforms.AutoAugment(
                    policy=tv.transforms.AutoAugmentPolicy.CIFAR10
                ),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    testset = tv.datasets.CIFAR10(
        root="./data/",
        train=False,
        transform=tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
elif "mnist" in trainname:
    trainset = tv.datasets.MNIST(
        "./data/",
        train=True,
        download=True,
        transform=tv.transforms.Compose(
            [
                tv.transforms.Resize(64),
                tv.transforms.Grayscale(3),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    testset = tv.datasets.MNIST(
        "./data/",
        train=False,
        download=True,
        transform=tv.transforms.Compose(
            [
                tv.transforms.Resize(64),
                tv.transforms.Grayscale(3),
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

# %%
def plot_results(
    train_loss_log, val_loss_log, epsilon_log, results, model_names, figure_suffix
):
    fig, axs = plt.subplots(1, 3 if len(epsilon_log) == 0 else 4)
    fig.set_figwidth(15)
    axs[0].plot(train_loss_log, label=model_names)
    axs[1].plot(val_loss_log, label=model_names)
    axs[2].scatter(
        [i for i in range(len(models))], [r["mcc"] for r in results], label="MCC"
    )
    axs[2].scatter(
        [i for i in range(len(models))], [r["acc"] for r in results], label="ACC"
    )
    axs[2].scatter(
        [i for i in range(len(models))], [r["rocauc"] for r in results], label="ROC-AUC"
    )
    if len(epsilon_log) > 0:
        axs[3].plot(epsilon_log)
        axs[3].set_title("privacy curve")
        axs[3].set_xlabel("Epoch")
        axs[3].set_ylabel("Epsilon value")
    axs[0].set_title("Train loss")
    axs[1].set_title("Val loss")
    axs[2].set_title("Test metrics")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[2].set_ylabel("Score accuracy")
    axs[0].set_xlabel("Train steps")
    axs[1].set_xlabel("Epochs")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[0].set_ylim([0, 5])
    axs[2].set_ylim([0, 1])
    axs[2].set_xticks(list(range(len(model_names))))
    axs[2].set_xticklabels(model_names)
    # for i, txt in enumerate(model_names):
    #     for key in ["mcc", "acc", "rocauc"]:
    #         axs[2].annotate(txt, (i, results[i][key]))
    fig.tight_layout(pad=3.0)
    plt.savefig(f"train_{figure_suffix}.png")


def to_one_hot(num_classes, targets):
    x = [[0 for _ in range(num_classes)] for _ in range(len(targets))]
    for i, t in enumerate(targets):
        x[i][t] = 1
    return x


def test(models, testloader, model_names):
    test_losses = [0 for _ in range(len(models))]
    preds, scores, targets = (
        [[] for _ in range(len(models))],
        [[] for _ in range(len(models))],
        [],
    )
    for batch_idx, (data, target) in tqdm(
        enumerate(testloader), total=len(testloader), desc="Testing", leave=False
    ):
        targets.extend(target.tolist())
        data, target = data.to(args.device), target.to(args.device)
        for model_idx in range(len(models)):
            with torch.no_grad():
                pred = models[model_idx](data)
                loss = loss_fn(pred, target)
                test_losses[model_idx] += loss.detach().cpu().item()
                scores[model_idx].extend(pred.detach().cpu().tolist())
                preds[model_idx].extend(
                    torch.argmax(pred.detach().cpu(), dim=1).tolist()
                )
    results = []
    for model_idx in range(len(models)):
        acc = metrics.accuracy_score(targets, preds[model_idx])
        mcc = metrics.matthews_corrcoef(targets, preds[model_idx])
        roc = metrics.roc_auc_score(
            to_one_hot(10, targets), scores[model_idx], multi_class="ovo"
        )
        results.append({"mcc": mcc, "acc": acc, "rocauc": roc})
    return results


def setup(args, deepee=False, opacus=False):
    model_names = ["VGG11", "VGG13", "VGG16", "VGG19"]
    models: list[torch.nn.Module] = [
        tv.models.vgg11_bn(pretrained=True),
        tv.models.vgg13_bn(pretrained=True),
        tv.models.vgg16_bn(pretrained=True),
        tv.models.vgg19_bn(pretrained=True),
    ]
    for i in range(len(models)):
        models[i].classifier = torch.nn.Linear(512 * 7 * 7, 10)
    if opacus:
        for i in range(len(models)):
            models[i] = module_modification.convert_batchnorm_modules(models[i])
    else:
        surgeon = ModelSurgeon(SurgicalProcedures.BN_to_GN)
        for i in range(len(models)):
            surgeon.operate(models[i])
    optims: list[torch.optim.Optimizer] = [
        torch.optim.Adam(models[i].parameters(), lr=args.lr) for i in range(len(models))
    ]
    for i in range(len(models)):
        models[i].to(args.device)
    if opacus:
        privacy_engines = [
            PrivacyEngine(
                models[i],
                noise_multiplier=args.noise_mult,
                max_grad_norm=args.clip_norm,
                target_delta=args.delta,
                batch_size=args.batch_size,
                sample_size=len(trainset),
                sample_rate=args.batch_size / len(trainset),
            )
            for i in range(len(models))
        ]
        for i in range(len(models)):
            privacy_engines[i].attach(optims[i])
            privacy_engines[i].to(args.device)
    scheds = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optims[i], T_max=args.num_epochs * len(trainloader)
        )
        for i in range(len(optims))
    ]
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    if deepee:
        watchdogs = [
            PrivacyWatchdog(
                trainloader,
                target_epsilon=200.0,
                abort=True,
                target_delta=args.delta,
                report_every_n_steps=len(trainloader),
                fallback_to_rdp=True,
            )
            for _ in models
        ]
        for i in range(len(models)):
            models[i] = PrivacyWrapper(
                models[i],
                args.batch_size,
                L2_clip=args.clip_norm,
                noise_multiplier=args.noise_mult,
                watchdog=watchdogs[i],  # watchdog,
            )
    return models, model_names, optims, scheds, loss_fn


def train(args, models, optims, scheds, loss_fn, deepee=False, opacus=False):
    # %%
    train_loss_log = []
    val_loss_log = []
    epsilon_log = []
    # %%
    # %%
    for epoch in tqdm(
        range(args.num_epochs),
        total=args.num_epochs,
        desc="Fitting networks to data",
        leave=False,
    ):
        for model in models:
            model.train()
        pbar = tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            desc=f"Training epoch {epoch}",
            leave=False,
        )
        for batch_idx, (data, target) in pbar:
            data, target = data.to(args.device), target.to(args.device)
            model_losses = [0 for _ in range(len(models))]
            for model_idx in range(len(models)):
                optims[model_idx].zero_grad()
                pred = models[model_idx](data)
                loss = loss_fn(pred, target)
                loss.backward()
                if deepee:
                    models[model_idx].clip_and_accumulate()
                    models[model_idx].noise_gradient()
                optims[model_idx].step()
                scheds[model_idx].step()
                if deepee:
                    models[model_idx].prepare_next_batch()
                model_losses[model_idx] += loss.detach().cpu().item() / data.shape[0]
            train_loss_log.append(model_losses)
            pbar.set_description(desc=f"Mean loss: {np.mean(model_losses):.3f}")
            if args.overfit and batch_idx > args.overfit:
                break
        for model in models:
            model.eval()
        val_losses = [0 for _ in range(len(models))]
        for batch_idx, (data, target) in tqdm(
            enumerate(valloader),
            total=len(valloader),
            desc=f"Validating epoch {epoch}",
            leave=False,
        ):
            data, target = data.to(args.device), target.to(args.device)
            for model_idx in range(len(models)):
                with torch.no_grad():
                    pred = models[model_idx](data)
                    loss = loss_fn(pred, target)
                    val_losses[model_idx] += loss.detach().cpu().item()
            if args.overfit and batch_idx > args.overfit:
                break
        for i in range(len(val_losses)):
            val_losses[i] /= len(trainset)
        val_loss_log.append(val_losses)
        if deepee:
            epsilon_log.append(models[0].current_epsilon)
        if opacus:
            epsilon_log.append(
                optims[0].privacy_engine.get_privacy_spent(args.delta)[0]
            )
        # print(val_losses)
    return train_loss_log, val_loss_log, epsilon_log


# %%
# Vanilla training
suffix = f"_vanilla_{trainname}"
models, model_names, optims, scheds, loss_fn = setup(args)
train_loss_log, val_loss_log, eps_log = train(args, models, optims, scheds, loss_fn)
results = test(models, testloader, model_names)
torch.save(
    {name: m.state_dict() for name, m in zip(model_names, models)},
    f"ablation_vggs{suffix}.pt",
)
torch.save({n: m for n, m in zip(model_names, results)}, f"mcc_results{suffix}.pt")
print("Vanilla results:")
for n, m in zip(model_names, results):
    print(f"\t- {n}:", end="\t")
    for key, value in m.items():
        print(f"{key}: {100.0*value:.1f}%", end="\t")
    print("")
# %%
plot_results(train_loss_log, val_loss_log, eps_log, results, model_names, suffix)


# # %%
# # %%
suffix = f"_deepee_{trainname}"
models, model_names, optims, scheds, loss_fn = setup(args, deepee=True)
train_loss_log, val_loss_log, eps_log = train(
    args, models, optims, scheds, loss_fn, deepee=True
)
results = test(models, testloader, model_names)
torch.save(
    {name: m.state_dict() for name, m in zip(model_names, models)},
    f"ablation_vggs{suffix}.pt",
)
torch.save({n: m for n, m in zip(model_names, results)}, f"mcc_results{suffix}.pt")
plot_results(train_loss_log, val_loss_log, eps_log, results, model_names, suffix)
print("Deepee results: ")
for n, m in zip(model_names, results):
    print(f"\t- {n}:", end="\t")
    for key, value in m.items():
        print(f"{key}: {100.0*value:.1f}%", end="\t")
    print("")
eps = [m.current_epsilon for m in models]
print(f"Epsila (should be identical): {eps}")

# %%
# opacus
suffix = f"_opacus_{trainname}"
models, model_names, optims, scheds, loss_fn = setup(args, opacus=True)
train_loss_log, val_loss_log, eps_log = train(
    args, models, optims, scheds, loss_fn, opacus=True
)
results = test(models, testloader, model_names)
torch.save(
    {name: m.state_dict() for name, m in zip(model_names, models)},
    f"ablation_vggs{suffix}.pt",
)
torch.save({n: m for n, m in zip(model_names, results)}, f"mcc_results{suffix}.pt")
plot_results(train_loss_log, val_loss_log, eps_log, results, model_names, suffix)
eps_alphas = [o.privacy_engine.get_privacy_spent(args.delta) for o in optims]
print("Opacus results: ")
for n, m in zip(model_names, results):
    print(f"\t- {n}:", end="\t")
    for key, value in m.items():
        print(f"{key}: {100.0*value:.1f}%", end="\t")
    print("")
print(f"Epsila (should be identical): {[i[0] for i in eps_alphas]}")

