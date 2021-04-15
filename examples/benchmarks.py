# %%
import torch
import torchvision as tv
from typing import Tuple, List

from time import time
from timeit import repeat

# %%
# common settings
class args:
    resolution: int = 256
    batch_size: int = 32
    secure_rng: bool = False  # true not supported yet
    steps: int = 25
    model: str = "resnet18"
    optim: str = "SGD"
    experiment: str = "memory"
    force_cpu: bool = experiment == "memory"
    framework: str = "deepee"
    task = "segmentation"
    if experiment == "memory":
        steps = 1
    assert framework in ["all", "deepee", "opacus", "pyvacy"]
    assert experiment in ["memory", "speed"]


# %%
class DPBenchmarkClassification:
    # @profile
    def __init__(self, args: args) -> None:
        self.args: args = args
        self.setup_model()
        self.setup_optim()
        self.setup_dataset()
        self.setup_dataloader()
        self.device: torch.device = torch.device(
            "cuda"
        ) if torch.cuda.is_available() and not args.force_cpu else torch.device("cpu")
        self.model.to(self.device)

    # @profile
    def __train_step__(self, data, target) -> None:
        self.on_before_train_step()
        self.optim.zero_grad()
        self.on_after_zero_grad()
        pred = self.make_prediction(data)
        self.on_after_prediction()
        loss = self.calc_loss(pred, target)
        self.on_after_loss()
        self.backward(loss)
        self.on_after_backward()
        self.optim.step()
        self.on_after_train_step()

    # @profile
    def time_training(self) -> float:
        tick = time()
        total_num_steps = 0
        self.on_before_training()
        while True:
            if total_num_steps > self.args.steps:  # meh
                break
            for data, target in self.trainloader:
                self.__train_step__(data.to(self.device), target.to(self.device))
                total_num_steps += 1
                if total_num_steps > self.args.steps:
                    break
        self.on_after_training()
        tock = time()
        return tock - tick

    def setup_model(self) -> None:
        if self.args.model == "resnet18":
            self.model: torch.nn.Module = tv.models.resnet18()
            self.model.fc = torch.nn.Linear(512, 1)
        elif self.args.model == "resnet152":
            self.model: torch.nn.Module = tv.models.resnet152()
            self.model.fc = torch.nn.Linear(2048, 1)
        elif self.args.model == "vgg11_bn":
            self.model: torch.nn.Module = tv.models.vgg11_bn()
            self.model.classifier = torch.nn.Linear(512 * 7 * 7, 1)
        else:
            raise ValueError(f"Model {self.args.model} not supported")

    def setup_optim(self) -> None:
        if self.args.optim == "SGD":
            self.optim: torch.optim.Optimizer = torch.optim.SGD(
                self.model.parameters(), lr=1.0
            )
        elif self.args.optim == "Adam":
            self.optim: torch.optim.Optimizer = torch.optim.Adam(
                self.model.parameters(), lr=1.0
            )
        else:
            raise ValueError(f"Optim {self.args.optim} not supported")

    def setup_dataset(self) -> None:
        self.dataset: torch.utils.data.Dataset = tv.datasets.FakeData(
            size=self.args.steps * self.args.batch_size,
            image_size=(3, self.args.resolution, self.args.resolution),
            num_classes=2,
            transform=tv.transforms.ToTensor(),
        )

    def setup_dataloader(self) -> None:
        if "dataset" not in self.__dict__.keys() or self.dataset is None:
            raise RuntimeError("Dataset was not initialised before dataloader")
        self.trainloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available() and not self.args.force_cpu,
            num_workers=0
            if torch.cuda.is_available() and not self.args.force_cpu
            else 8,
        )

    def make_prediction(self, data: torch.tensor) -> torch.tensor:
        x: torch.tensor = self.model(data)
        x.squeeze_(1)
        return x

    def calc_loss(self, pred: torch.tensor, target: torch.tensor) -> torch.tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            pred, target.to(torch.float)
        )

    def backward(self, loss: torch.tensor) -> None:
        loss.backward()

    def on_before_training(self) -> None:
        pass

    def on_after_training(self) -> None:
        pass

    def on_before_train_step(self) -> None:
        pass

    def on_after_train_step(self) -> None:
        pass

    def on_after_zero_grad(self) -> None:
        pass

    def on_after_prediction(self) -> None:
        pass

    def on_after_loss(self) -> None:
        pass

    def on_after_backward(self) -> None:
        pass


# %%
import segmentation_models_pytorch as smp


class DPBenchmarkAutoencoder(DPBenchmarkClassification):
    def __init__(self, args):
        super().__init__(args)
        self.loss_fn = smp.utils.losses.DiceLoss()

    def setup_model(self):
        if self.args.model == "resnet18":
            self.model = smp.Unet(
                encoder_name="vgg11_bn",
                encoder_weights="imagenet",
                classes=1,
                in_channels=1,
                activation="sigmoid",
            )
        else:
            raise ValueError(f"Model {self.args.model} not supported")

    def setup_dataset(self) -> None:
        self.dataset: torch.utils.data.Dataset = tv.datasets.FakeData(
            size=self.args.steps * self.args.batch_size,
            image_size=(1, self.args.resolution, self.args.resolution),
            num_classes=2,
            transform=tv.transforms.ToTensor(),
        )

    def __train_step__(self, data, target) -> None:
        self.on_before_train_step()
        self.optim.zero_grad()
        self.on_after_zero_grad()
        pred = self.make_prediction(data)
        self.on_after_prediction()
        loss = self.calc_loss(pred, data)
        self.on_after_loss()
        self.backward(loss)
        self.on_after_backward()
        self.optim.step()
        self.on_after_train_step()

    def make_prediction(self, data):
        return self.model(data)

    def calc_loss(self, pred, target):
        return self.loss_fn(pred, target)


# %%
def analyse_result(timeit_result: List[float]) -> str:
    sorted_list = sorted(timeit_result)
    m = float(sum(timeit_result)) / len(timeit_result)
    return (
        f"Timing results:\n\t"
        f"- Min: {sorted_list[0]:.2f}s\n\t"
        f"- Max: {sorted_list[-1]:.2f}s\n\t"
        f"- Mean: {m:.2f}s\n\t"
        f"- Median: {sorted_list[len(sorted_list)//2]:.2f}s"
    )


# %%
# 1) deepee
if args.framework in ["all", "deepee"]:
    from deepee.watchdog import PrivacyWatchdog
    from deepee import PrivacyWrapper
    from deepee import ModelSurgeon, SurgicalProcedures
    from deepee import UniformDataLoader

    class deepee_DPTrainer(
        DPBenchmarkAutoencoder
        if args.task == "segmentation"
        else DPBenchmarkClassification
    ):
        # class deepee_DPTrainer(DPBenchmarkClassification):
        # @profile
        def __init__(self, args: args):
            super().__init__(args)
            watchdog = PrivacyWatchdog(
                self.trainloader,
                target_epsilon=1e9,
                abort=True,
                target_delta=1e-5,
                report_every_n_steps=1e9,
            )
            surgeon = ModelSurgeon(SurgicalProcedures.BN_to_BN_nostats)
            surgeon.operate(self.model)
            self.model = PrivacyWrapper(
                self.model,
                self.args.batch_size,
                1.0,
                1.0,
                watchdog=watchdog,
                secure_rng=args.secure_rng,
            ).to(self.device)

        def setup_dataloader(self) -> None:
            self.trainloader = UniformDataLoader(
                self.dataset,
                self.args.batch_size,
                pin_memory=torch.cuda.is_available() and not self.args.force_cpu,
                num_workers=0
                if torch.cuda.is_available() and not self.args.force_cpu
                else 8,
            )

        def on_after_backward(self) -> None:
            self.model.clip_and_accumulate()
            self.model.noise_gradient()

        def on_after_train_step(self) -> None:
            self.model.prepare_next_batch()


# %%
# opacus
if args.framework in ["all", "opacus"]:
    import opacus
    from opacus.utils.module_modification import convert_batchnorm_modules
    from opacus.utils.uniform_sampler import UniformWithReplacementSampler

    class opacus_DPTrainer(
        DPBenchmarkAutoencoder
        if args.task == "segmentation"
        else DPBenchmarkClassification
    ):
        # class opacus_DPTrainer(DPBenchmarkClassification):
        def __init__(self, args):
            super().__init__(args)
            self.model = convert_batchnorm_modules(self.model.cpu()).to(self.device)
            self.setup_optim()
            privacy_engine = opacus.PrivacyEngine(
                self.model,
                sample_rate=self.args.batch_size / len(self.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                secure_rng=args.secure_rng,
            )
            privacy_engine.attach(self.optim)


# %%
if args.framework in ["all", "pyvacy"]:
    from pyvacy import optim as pyvacyoptim
    from pyvacy import sampling as pyvacysampling

    class pyvacy_DPTrainer(
        DPBenchmarkAutoencoder
        if args.task == "segmentation"
        else DPBenchmarkClassification
    ):
        # class pyvacy_DPTrainer(DPBenchmarkClassification):
        def __init__(self, args):
            super().__init__(args)
            self.setup_optim()
            if args.secure_rng:
                raise ValueError(f"Secure rng not supported in pyvacy")

        def setup_optim(self):
            self.optim = pyvacyoptim.DPSGD(
                params=self.model.parameters(),
                lr=1.0,
                l2_norm_clip=1.0,
                noise_multiplier=1.0,
                minibatch_size=args.batch_size,
                microbatch_size=1,
            )

        def setup_dataloader(self):
            minibatch_loader, self.microbatchloader = pyvacysampling.get_data_loaders(
                minibatch_size=self.args.batch_size,
                microbatch_size=1,
                iterations=self.args.steps,
            )
            self.trainloader = minibatch_loader(self.dataset)

        # @profile
        def __train_step__(self, data, target) -> None:
            self.optim.zero_grad()
            for micro_x, micro_y in self.microbatchloader(
                torch.utils.data.TensorDataset(data, target)
            ):
                self.optim.zero_microbatch_grad()
                pred = self.make_prediction(micro_x)
                if self.args == "segmentation":
                    loss = self.calc_loss(pred, micro_x)  # different for classification
                else:
                    loss = self.calc_loss(pred, micro_y)
                self.backward(loss)
                self.optim.microbatch_step()
            self.optim.step()


# # %%
# from gradcnn import make_optimizer, replicate_model
# from gradcnn import crb as nn

# # %%
# class owkin_DPTrainer(DPTraining):
#     def __init__(self, args):
#         super().__init__(args)

#     def setup_model(self):
#         pass


# %%
def deepee_run(num_runs):
    deepee_tr = deepee_DPTrainer(args)
    deepee_t = repeat(deepee_tr.time_training, number=num_runs)
    print(
        f"Results for deepee [n={num_runs}, steps={args.steps}]:\n{analyse_result(deepee_t)}\n"
    )


def opacus_run(num_runs):
    opacus_tr = opacus_DPTrainer(args)
    opacus_t = repeat(opacus_tr.time_training, number=num_runs)
    print(
        f"Results for opacus [n={num_runs}, steps={args.steps}]:\n{analyse_result(opacus_t)}\n"
    )


def pyvacy_run(num_runs):
    pyvacy_tr = pyvacy_DPTrainer(args)
    pyvacy_t = repeat(pyvacy_tr.time_training, number=num_runs)
    print(
        f"Results for pyvacy [n={num_runs}, steps={args.steps}]:\n{analyse_result(pyvacy_t)}\n"
    )


if args.experiment == "memory":

    @profile
    def profile_memory():
        if args.framework == "deepee":
            tr = deepee_DPTrainer(args)
        elif args.framework == "opacus":
            tr = opacus_DPTrainer(args)
        elif args.framework == "pyvacy":
            tr = pyvacy_DPTrainer(args)
        else:
            raise ValueError(
                f"For memory profiling framework needs to be defined as either deepee,"
                f" opacus or pyvacy. [Value: {args.framework}]"
            )
        tr.time_training()

    profile_memory()
elif args.experiment == "speed":
    n = 5
    if args.framework in ["deepee", "all"]:
        try:
            deepee_run(n)
        except Exception as e:
            print(f"Deepee failed: {str(e)}")
    if args.framework in ["opacus", "all"]:
        try:
            opacus_run(n)
        except Exception as e:
            print(f"Deepee failed: {str(e)}")
    if args.framework in ["pyvacy", "all"]:
        try:
            pyvacy_run(n)
        except Exception as e:
            print(f"Deepee failed: {str(e)}")
