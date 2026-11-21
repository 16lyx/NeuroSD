import csv
import hashlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import torch


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())

    if isinstance(value, float) and not np.isfinite(value):
        return str(value)

    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return str(value)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(data),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.exists() and path.stat().st_size > 0
    fields = list(row)

    if exists:
        with path.open(newline="", encoding="utf-8") as file:
            previous = next(csv.reader(file))

        if previous != fields:
            raise RuntimeError(
                f"CSV columns do not match: {path}"
            )

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        if not exists:
            writer.writeheader()

        writer.writerow(json_safe(row))


def replace_csv(path, rows):
    rows = list(rows)

    if not rows:
        return

    fields = list(rows[0])

    for row in rows:
        if set(row) != set(fields):
            raise ValueError(
                "All CSV rows must contain the same fields."
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            json_safe(row) for row in rows
        )


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def command_output(command, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as error:
        return {"error": str(error)}


def memory_now(device):
    result = {
        "process_tree_rss_mb": None,
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
        "cuda_peak_allocated_mb": None,
        "cuda_peak_reserved_mb": None,
    }

    try:
        import psutil

        process = psutil.Process()
        rss = process.memory_info().rss

        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        result["process_tree_rss_mb"] = rss / 1024**2
    except ImportError:
        pass

    device = torch.device(device)

    if device.type == "cuda":
        result.update(
            {
                "cuda_allocated_mb": (
                    torch.cuda.memory_allocated(device) / 1024**2
                ),
                "cuda_reserved_mb": (
                    torch.cuda.memory_reserved(device) / 1024**2
                ),
                "cuda_peak_allocated_mb": (
                    torch.cuda.max_memory_allocated(device)
                    / 1024**2
                ),
                "cuda_peak_reserved_mb": (
                    torch.cuda.max_memory_reserved(device)
                    / 1024**2
                ),
            }
        )

    return result


def sync_device(device):
    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_gpu_peak(device):
    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def parameter_records(model):
    def count(parameters):
        parameters = list(parameters)

        return {
            "tensor_elements": sum(
                parameter.numel()
                for parameter in parameters
            ),
            "real_dof": sum(
                parameter.numel()
                * (2 if parameter.is_complex() else 1)
                for parameter in parameters
            ),
        }

    return {
        "all": count(model.parameters()),
        "trainable": count(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "by_top_level_module": {
            name: count(module.parameters())
            for name, module in model.named_children()
        },
        "note": (
            "Each complex tensor element represents two real "
            "scalar degrees of freedom. Module counts may overlap "
            "when parameters are shared."
        ),
    }


def unwrap_loader(loader):
    seen = set()

    while hasattr(loader, "loader"):
        if id(loader) in seen:
            raise RuntimeError(
                "A cyclic loader wrapper was detected."
            )

        seen.add(id(loader))
        loader = loader.loader

    return loader


def loader_record(loader):
    base_loader = unwrap_loader(loader)
    dataset = loader.dataset
    fixed = getattr(dataset, "dataset", None)

    def shape_of(value):
        shape = getattr(value, "shape", None)
        return list(shape) if shape is not None else None

    online_generated = None

    if hasattr(dataset, "is_train"):
        online_generated = bool(dataset.is_train)
    elif hasattr(dataset, "X"):
        online_generated = False

    return {
        "dataset_class": type(dataset).__name__,
        "dataset_length": len(dataset),
        "batches_per_epoch": len(loader),
        "batch_size": getattr(base_loader, "batch_size", None),
        "drop_last": getattr(base_loader, "drop_last", None),
        "num_workers": getattr(base_loader, "num_workers", None),
        "persistent_workers": getattr(
            base_loader,
            "persistent_workers",
            None,
        ),
        "sampler_class": type(base_loader.sampler).__name__,
        "is_train": getattr(dataset, "is_train", None),
        "online_generated": online_generated,
        "fixed_array_shape": shape_of(fixed),
        "mean": getattr(dataset, "mean", None),
        "std": getattr(dataset, "std", None),
        "input_array_shape": shape_of(
            getattr(dataset, "X", None)
        ),
        "target_array_shape": shape_of(
            getattr(dataset, "Y", None)
        ),
        "use_augment": getattr(dataset, "use_augment", None),
        "stored_data_shape": shape_of(
            getattr(dataset, "data", None)
        ),
        "trajectory_ids": getattr(
            dataset,
            "trajectory_ids",
            None,
        ),
        "normalization": getattr(
            dataset,
            "normalization",
            None,
        ),
        "source_file": getattr(dataset, "file_path", None),
        "source_axis_order": getattr(dataset, "axis_order", None),
    }


def save_experiment_metadata(exp, project_root):
    root = Path(project_root).resolve()
    output = Path(exp.path)
    output.mkdir(parents=True, exist_ok=True)

    packages = {}

    for name in (
        "torch",
        "torchvision",
        "numpy",
        "timm",
        "torchdiffeq",
        "pytorch-msssim",
        "scikit-image",
        "fvcore",
        "psutil",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None

    ram_gb = None

    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / 1024**3
    except ImportError:
        pass

    gpus = []

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)

            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gb": (
                        properties.total_memory / 1024**3
                    ),
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                    "multiprocessor_count": (
                        properties.multi_processor_count
                    ),
                }
            )

    environment = {
        "utc_start_time": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": platform.processor(),
        "cpu_logical_count": os.cpu_count(),
        "host_ram_gb": ram_gb,
        "packages": packages,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "actual_device": str(exp.device),
        "visible_gpus": gpus,
        "nvidia_smi": command_output(["nvidia-smi"]),
        "lscpu": command_output(["lscpu"]),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": (
            torch.get_num_interop_threads()
        ),
        "cudnn_deterministic": (
            torch.backends.cudnn.deterministic
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "environment_variables": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "git_head": command_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
        ),
        "git_status": command_output(
            ["git", "status", "--short"],
            cwd=root,
        ),
    }

    write_json(output / "environment.json", environment)
    write_json(
        output / "config_resolved.json",
        vars(exp.args),
    )
    write_json(
        output / "parameters.json",
        parameter_records(exp.method.model),
    )

    optimizer = exp.method.model_optim
    scheduler = exp.method.scheduler

    parameter_dtypes = sorted(
        {
            str(parameter.dtype)
            for parameter in exp.method.model.parameters()
        }
    )

    write_json(
        output / "optimizer_scheduler.json",
        {
            "optimizer_class": type(optimizer).__name__,
            "optimizer_defaults": optimizer.defaults,
            "parameter_groups": [
                {
                    key: value
                    for key, value in group.items()
                    if key != "params"
                }
                for group in optimizer.param_groups
            ],
            "scheduler_class": (
                type(scheduler).__name__
                if scheduler is not None
                else None
            ),
            "scheduler_initial_state": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "scheduler_by_epoch": exp.method.by_epoch,
            "gradient_clip_norm": getattr(
                exp.args,
                "clip_grad",
                None,
            ),
            "gradient_clip_mode": getattr(
                exp.args,
                "clip_mode",
                None,
            ),
            "training_precision": (
                "mixed_float16"
                if getattr(exp.args, "fp16", False)
                else "float32"
            ),
            "model_parameter_dtypes": parameter_dtypes,
        },
    )

    shared = exp.vali_loader.dataset is exp.test_loader.dataset
    dataname = exp.args.dataname

    value_scales = {
        "mmnist": "pixel values divided by 255",
        "taxibj": "(stored values + 1) / 2",
        "navier_stokes_t20": (
            "normalized vorticity; see dataset normalization"
        ),
        "navier_stokes_t30": (
            "normalized vorticity; see dataset normalization"
        ),
        "navier_stokes_t50": (
            "normalized vorticity; see dataset normalization"
        ),
    }

    write_json(
        output / "data_protocol.json",
        {
            "train": loader_record(exp.train_loader),
            "validation": loader_record(exp.vali_loader),
            "test": loader_record(exp.test_loader),
            "validation_test_same_dataset_object": shared,
            "independent_final_test": False if shared else None,
            "independence_note": (
                "Validation and test use the same dataset object."
                if shared
                else (
                    "Different dataset objects alone do not prove "
                    "that the underlying data are disjoint."
                )
            ),
            "metric_protocol": {
                "value_scale": value_scales.get(
                    dataname,
                    "See the dataset preprocessing implementation.",
                ),
                "mse_mae": "Unclipped predictions.",
                "mse_reported": (
                    "Mean elementwise squared error multiplied "
                    "by C*H*W."
                ),
                "mae_reported": (
                    "Mean elementwise absolute error multiplied "
                    "by C*H*W."
                ),
                "ssim": (
                    "skimage; window size 7; data_range=1; "
                    "mean over channels, samples and frames."
                ),
                "psnr": (
                    "Mean per-frame PSNR; data_range=1; "
                    "frame MSE averaged over C*H*W."
                ),
                "ssim_psnr_prediction_clipping": "[0,1]",
                "ground_truth_clipping": "none",
                "training_ssim": (
                    "pytorch_msssim.ssim; data_range=1; "
                    "unclipped predictions."
                ),
            },
            "selection_metric": exp.args.selection_metric,
            "result_source": "this training run",
            "research_notes": getattr(
                exp.args,
                "research_notes",
                {},
            ),
        },
    )

    candidates = [
        root / "openstl/models/neurosd_model.py",
        root / "openstl/methods/neurosd.py",
        root / "openstl/utils/experiment_log.py",
        root / "openstl/datasets/utils.py",
        root / "openstl/datasets/dataloader.py",
        root / "openstl/datasets/dataset_constant.py",
        root / "openstl/core/optim_scheduler.py",
        root / "tools/train.py",
        root / "openstl/methods/baseline_logging.py",
    ]

    loader_files = {
        "mmnist": "dataloader_moving_mnist.py",
        "taxibj": "dataloader_taxibj.py",
        "navier_stokes_t20": "dataloader_navier_stokes_t20.py",
        "navier_stokes_t30": "dataloader_navier_stokes_t30.py",
        "navier_stokes_t50": "dataloader_navier_stokes_t50.py",
    }

    if dataname in loader_files:
        candidates.append(
            root / "openstl/datasets" / loader_files[dataname]
        )

    config_file = getattr(exp.args, "config_file", None)

    if config_file:
        candidates.append(Path(config_file))

    data_files = {
        "mmnist": [
            "moving_mnist/train-images-idx3-ubyte.gz",
            "moving_mnist/mnist_test_seq.npy",
        ],
        "taxibj": [
            "taxibj/dataset.npz",
        ],
    }

    candidates.extend(
        Path(exp.args.data_root) / relative_path
        for relative_path in data_files.get(dataname, [])
    )

    for loader in (
        exp.train_loader,
        exp.vali_loader,
        exp.test_loader,
    ):
        source_file = getattr(
            loader.dataset,
            "file_path",
            None,
        )

        if source_file:
            candidates.append(Path(source_file))

    ns_file = getattr(exp.args, "ns_data_file", None)

    if ns_file:
        candidates.append(Path(ns_file))

    for instance in (exp.method, exp.method.model):
        try:
            candidates.append(
                Path(inspect.getfile(type(instance)))
            )
        except (TypeError, OSError):
            pass

    files = {}

    for candidate in candidates:
        path = candidate

        if not path.is_absolute():
            if not path.is_file():
                path = root / path

        if not path.is_file():
            continue

        path = path.resolve()
        key = str(path)

        if key in files:
            continue

        files[key] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    write_json(output / "file_manifest.json", files)

    (output / "model_structure.txt").write_text(
        str(exp.method.model),
        encoding="utf-8",
    )

    frozen = command_output(
        [sys.executable, "-m", "pip", "freeze"]
    )

    (output / "pip_freeze.txt").write_text(
        frozen.get("stdout", str(frozen)),
        encoding="utf-8",
    )


@torch.no_grad()
def profile_model(exp):
    model = exp.method.model

    if getattr(exp.method, "_baseline_logging", False):
        from openstl.methods.baseline_logging import ForecastView

        model = ForecastView(exp.method)

    device = torch.device(exp.device)

    modules = list(model.modules())

    if exp.method.model is not model:
        modules.extend(exp.method.model.modules())

    module_states = {
        id(module): (module, module.training)
        for module in modules
    }

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()

    cuda_rng = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    try:
        model.eval()
        exp.method.model.eval()

        parameter = next(
            (
                parameter
                for parameter in exp.method.model.parameters()
                if parameter.is_floating_point()
            ),
            None,
        )

        input_dtype = (
            parameter.dtype
            if parameter is not None
            else torch.float32
        )

        inputs = torch.zeros(
            1,
            exp.args.pre_seq_length,
            *exp.args.in_shape[1:],
            device=device,
            dtype=input_dtype,
        )

        warmup = int(exp.args.profile_warmup)
        repeats = int(exp.args.profile_repeats)

        if warmup < 0:
            raise ValueError(
                "profile_warmup must be nonnegative."
            )

        if repeats <= 0:
            raise ValueError(
                "profile_repeats must be positive."
            )

        for _ in range(warmup):
            model(inputs)

        sync_device(device)
        reset_gpu_peak(device)

        baseline_memory = memory_now(device)
        elapsed_ms = []

        for _ in range(repeats):
            sync_device(device)
            start = time.perf_counter()

            model(inputs)

            sync_device(device)
            elapsed_ms.append(
                (time.perf_counter() - start) * 1000.0
            )

        after_memory = memory_now(device)
        mean_latency = float(np.mean(elapsed_ms))

        forward_nfe = getattr(model, "nfe", None)

        if forward_nfe is None:
            forward_nfe = getattr(
                exp.method.model,
                "nfe",
                None,
            )

        efficiency = {
            "batch_size": 1,
            "input_shape": list(inputs.shape),
            "output_frames": exp.args.aft_seq_length,
            "warmup_iterations": warmup,
            "measured_iterations": repeats,
            "precision": str(input_dtype).replace("torch.", ""),
            "latency_ms_mean": mean_latency,
            "latency_ms_std": (
                float(np.std(elapsed_ms, ddof=1))
                if repeats > 1
                else None
            ),
            "latency_ms_median": float(np.median(elapsed_ms)),
            "latency_ms_p95": float(
                np.percentile(elapsed_ms, 95)
            ),
            "sequences_per_second": (
                1000.0 / mean_latency
            ),
            "predicted_frames_per_second": (
                1000.0
                * exp.args.aft_seq_length
                / mean_latency
            ),
            "forward_nfe": forward_nfe,
            "memory_before_timed_forward": baseline_memory,
            "memory_after_timed_forward": after_memory,
            "memory_unit": "MiB; legacy field names use '_mb'.",
            "cpu_memory_scope": (
                "Sampled process-tree RSS, not a stage peak."
            ),
            "timing_scope": (
                "Model forward only; excludes data loading, "
                "host-device transfer and metric calculation."
            ),
            "checkpoint_note": (
                "Profiles the currently loaded weights. "
                "This function does not load a checkpoint."
            ),
        }

        baseline_allocated = baseline_memory[
            "cuda_allocated_mb"
        ]
        peak_allocated = after_memory[
            "cuda_peak_allocated_mb"
        ]

        efficiency["incremental_peak_allocated_mb"] = (
            peak_allocated - baseline_allocated
            if (
                peak_allocated is not None
                and baseline_allocated is not None
            )
            else None
        )

        write_json(
            Path(exp.path) / "efficiency.json",
            efficiency,
        )

        counts = {
            "tool": "fvcore.FlopCountAnalysis",
            "input_shape": list(inputs.shape),
            "convention": (
                "One fused multiply-add counts as one operation."
            ),
            "complete_model_flops": None,
            "note": (
                "Supported operations only. FFT, complex arithmetic, "
                "solver operations and other unsupported operations "
                "require a separate audit before reporting "
                "complete-model FLOPs."
            ),
        }

        try:
            from fvcore.nn import FlopCountAnalysis

            analysis = FlopCountAnalysis(model, inputs)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)

            total = float(analysis.total())

            counts.update(
                {
                    "supported_operations": total,
                    "supported_giga_operations": total / 1e9,
                    "by_operator": dict(analysis.by_operator()),
                    "by_module": dict(analysis.by_module()),
                    "unsupported_operations": dict(
                        analysis.unsupported_ops()
                    ),
                    "uncalled_modules": sorted(
                        analysis.uncalled_modules()
                    ),
                }
            )
        except Exception as error:
            counts["error"] = repr(error)

        write_json(
            Path(exp.path) / "operation_count.json",
            counts,
        )

        return efficiency

    finally:
        for module, training in module_states.values():
            module.training = training

        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)

        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)