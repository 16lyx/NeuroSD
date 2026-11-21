def load_data(
    dataname,
    batch_size,
    val_batch_size,
    num_workers,
    data_root,
    dist=False,
    **kwargs,
):
    default_length = 4 if dataname == "taxibj" else 10

    pre_seq_length = kwargs.get("pre_seq_length")
    aft_seq_length = kwargs.get("aft_seq_length")

    if pre_seq_length is None:
        pre_seq_length = default_length

    if aft_seq_length is None:
        aft_seq_length = default_length

    cfg_dataloader = {
        "pre_seq_length": pre_seq_length,
        "aft_seq_length": aft_seq_length,
        "in_shape": kwargs.get("in_shape"),
        "distributed": dist,
        "use_augment": kwargs.get("use_augment", False),
        "use_prefetcher": kwargs.get("use_prefetcher", False),
        "drop_last": kwargs.get("drop_last", False),
    }

    if dataname == "bair":
        from .dataloader_bair import load_data as dataset_loader

    elif dataname == "human":
        from .dataloader_human import load_data as dataset_loader

    elif dataname == "kitticaltech":
        from .dataloader_kitticaltech import load_data as dataset_loader

    elif "kth" in dataname:
        from .dataloader_kth import load_data as dataset_loader

    elif dataname in ("mmnist", "mfmnist", "mmnist_cifar"):
        from .dataloader_moving_mnist import load_data as dataset_loader

        cfg_dataloader["data_name"] = kwargs.get(
            "data_name", "mnist"
        )

    elif "noisymmnist" in dataname:
        from .dataloader_noisy_moving_mnist import (
            load_data as dataset_loader,
        )

        cfg_dataloader["noise_type"] = kwargs.get(
            "noise_type", "perceptual"
        )

    elif "kinetics" in dataname:
        from .dataloader_kinetics import load_data as dataset_loader

        cfg_dataloader["data_name"] = kwargs.get(
            "data_name", "kinetics400"
        )

    elif dataname == "taxibj":
        from .dataloader_taxibj import load_data as dataset_loader

    elif "weather" in dataname:
        from .dataloader_weather import load_data as dataset_loader

        data_split = "5_625"

        for candidate in ("5_625", "2_8125", "1_40625"):
            if candidate in dataname:
                data_split = candidate

        weather_kwargs = dict(kwargs)
        weather_kwargs["distributed"] = dist
        weather_kwargs["data_split"] = data_split

        return dataset_loader(
            batch_size,
            val_batch_size,
            data_root,
            num_workers,
            **weather_kwargs,
        )

    elif dataname == "navier_stokes_t20":
        from .dataloader_navier_stokes_t20 import (
            load_data as dataset_loader,
        )

        cfg_dataloader["data_file"] = kwargs.get("ns_data_file")
        cfg_dataloader["axis_order"] = kwargs.get("ns_axis_order")

    elif dataname == "navier_stokes_t30":
        from .dataloader_navier_stokes_t30 import (
            load_data as dataset_loader,
        )

        cfg_dataloader["data_file"] = kwargs.get("ns_data_file")
        cfg_dataloader["axis_order"] = kwargs.get("ns_axis_order")

    elif dataname == "navier_stokes_t50":
        from .dataloader_navier_stokes_t50 import (
            load_data as dataset_loader,
        )

        cfg_dataloader["normalization_frames"] = kwargs.get(
            "ns_normalization_frames"
        )
        cfg_dataloader["data_file"] = kwargs.get("ns_data_file")
        cfg_dataloader["axis_order"] = kwargs.get("ns_axis_order")

    else:
        raise ValueError(
            f"Unsupported dataset: {dataname}"
        )

    return dataset_loader(
        batch_size,
        val_batch_size,
        data_root,
        num_workers,
        **cfg_dataloader,
    )