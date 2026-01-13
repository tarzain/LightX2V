import json
import os

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.tensor.device_mesh import init_device_mesh

from lightx2v.utils.input_info import ALL_INPUT_INFO_KEYS
from lightx2v.utils.lockable_dict import LockableDict
from lightx2v_platform.base.global_var import AI_DEVICE


def get_default_config():
    default_config = {
        "do_mm_calib": False,
        "cpu_offload": False,
        "max_area": False,
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "feature_caching": "NoCaching",  # ["NoCaching", "TaylorSeer", "Tea"]
        "teacache_thresh": 0.26,
        "use_ret_steps": False,
        "use_bfloat16": True,
        "lora_configs": None,  # List of dicts with 'path' and 'strength' keys
        "use_prompt_enhancer": False,
        "parallel": False,
        "seq_parallel": False,
        "cfg_parallel": False,
        "enable_cfg": False,
        "use_image_encoder": True,
    }
    default_config = LockableDict(default_config)
    return default_config


def set_config(args):
    config = get_default_config()
    config.update({k: v for k, v in vars(args).items() if k not in ALL_INPUT_INFO_KEYS})

    if config.get("config_json", None) is not None:
        logger.info(f"Loading some config from {config['config_json']}")
        with open(config["config_json"], "r") as f:
            config_json = json.load(f)
        config.update(config_json)

    if config["model_cls"] in ["hunyuan_video_1.5", "hunyuan_video_1.5_distill"]:  # Special config for hunyuan video 1.5 model folder structure
        config["transformer_model_path"] = os.path.join(config["model_path"], "transformer", config["transformer_model_name"])  # transformer_model_name: [480p_t2v, 480p_i2v, 720p_t2v, 720p_i2v]
        if os.path.exists(os.path.join(config["transformer_model_path"], "config.json")):
            with open(os.path.join(config["transformer_model_path"], "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
    else:
        if os.path.exists(os.path.join(config["model_path"], "config.json")):
            with open(os.path.join(config["model_path"], "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
        elif os.path.exists(os.path.join(config["model_path"], "low_noise_model", "config.json")):  # 需要一个更优雅的update方法
            with open(os.path.join(config["model_path"], "low_noise_model", "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
        elif os.path.exists(os.path.join(config["model_path"], "distill_models", "low_noise_model", "config.json")):  # 需要一个更优雅的update方法
            with open(os.path.join(config["model_path"], "distill_models", "low_noise_model", "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
        elif os.path.exists(os.path.join(config["model_path"], "original", "config.json")):
            with open(os.path.join(config["model_path"], "original", "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
        elif os.path.exists(os.path.join(config["model_path"], "transformer", "config.json")):
            with open(os.path.join(config["model_path"], "transformer", "config.json"), "r") as f:
                model_config = json.load(f)
            config.update(model_config)
        # load quantized config
        if config.get("dit_quantized_ckpt", None) is not None:
            config_path = os.path.join(config["dit_quantized_ckpt"], "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    model_config = json.load(f)
                config.update(model_config)

    # Some model configs use scalar ints for fields that LightX2V expects to be (t, h, w) tuples.
    # Normalize here to avoid runtime failures like "'int' object is not subscriptable".
    _default_config = get_default_config()

    def _normalize_triplet(value, default_triplet):
        if isinstance(value, int):
            return (value, default_triplet[1], default_triplet[2])
        if isinstance(value, (list, tuple)):
            if len(value) == 3:
                return tuple(value)
            if len(value) == 1:
                return (value[0], default_triplet[1], default_triplet[2])
        return default_triplet

    config["vae_stride"] = _normalize_triplet(config.get("vae_stride"), _default_config["vae_stride"])
    # For LTX2, some configs use scalar `patch_size=2` to mean spatial patching of (1, 2, 2).
    patch_size = config.get("patch_size")
    if isinstance(patch_size, int):
        config["patch_size"] = (1, patch_size, patch_size)
    else:
        config["patch_size"] = _normalize_triplet(patch_size, _default_config["patch_size"])

    if config["task"] in ["i2v", "s2v"]:
        if config["target_video_length"] % config["vae_stride"][0] != 1:
            logger.warning(f"`num_frames - 1` has to be divisible by {config['vae_stride'][0]}. Rounding to the nearest number.")
            config["target_video_length"] = config["target_video_length"] // config["vae_stride"][0] * config["vae_stride"][0] + 1

    if config["task"] not in ["t2i", "i2i"] and config["model_cls"] not in ["hunyuan_video_1.5", "hunyuan_video_1.5_distill"]:
        config["attnmap_frame_num"] = ((config["target_video_length"] - 1) // config["vae_stride"][0] + 1) // config["patch_size"][0]
        if config["model_cls"] in ["seko_talk", "wan2.2_animate"]:
            config["attnmap_frame_num"] += 1
            config["padding_multiple"] = config["attnmap_frame_num"]

    # Load diffusers vae config
    if os.path.exists(os.path.join(config["model_path"], "vae", "config.json")):
        with open(os.path.join(config["model_path"], "vae", "config.json"), "r") as f:
            vae_config = json.load(f)
            if "temperal_downsample" in vae_config:
                vae_config["vae_scale_factor"] = 2 ** len(vae_config["temperal_downsample"])
            elif "block_out_channels" in vae_config:
                vae_config["vae_scale_factor"] = 2 ** (len(vae_config["block_out_channels"]) - 1)
        config.update(vae_config)

    return config


def set_parallel_config(config):
    parallel_cfg = config.get("parallel")
    # Some config JSONs include a `parallel` dict even for single-process runs. Only enable
    # distributed parallelism when the requested mesh size is > 1 and torch.distributed is initialized.
    if not isinstance(parallel_cfg, dict):
        return

    cfg_p_size = parallel_cfg.get("cfg_p_size", 1)
    seq_p_size = parallel_cfg.get("seq_p_size", 1)
    if cfg_p_size * seq_p_size <= 1:
        return

    if not dist.is_available() or not dist.is_initialized():
        raise ValueError("Parallel config requests distributed execution, but torch.distributed is not initialized.")

    assert cfg_p_size * seq_p_size == dist.get_world_size(), f"cfg_p_size * seq_p_size must be equal to world_size"
    config["device_mesh"] = init_device_mesh(AI_DEVICE, (cfg_p_size, seq_p_size), mesh_dim_names=("cfg_p", "seq_p"))

    if parallel_cfg.get("seq_p_size", 1) > 1:
        config["seq_parallel"] = True

    if config.get("enable_cfg", False) and parallel_cfg.get("cfg_p_size", 1) > 1:
        config["cfg_parallel"] = True

    # warmup dist
    _a = torch.zeros([1]).to(f"{AI_DEVICE}:{dist.get_rank()}")
    dist.all_reduce(_a)


def print_config(config):
    config_to_print = config.copy()
    config_to_print.pop("device_mesh", None)
    # Don't assume torch.distributed is initialized just because `parallel` exists in config.
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")
    else:
        logger.info(f"config:\n{json.dumps(config_to_print, ensure_ascii=False, indent=4)}")
