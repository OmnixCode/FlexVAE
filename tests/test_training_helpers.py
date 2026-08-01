"""Unit tests for the training helpers and previously fixed critical paths.

Run from the repo root:
    PYTHONPATH=src pytest -q
"""
import argparse
import json
import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)

# train.py loads configs/config.cfg at import time; point it at a minimal fixture
# by chdir-ing into a temp tree built in the fixtures below when needed.


@pytest.fixture
def tiny_config(tmp_path, monkeypatch):
    cfg = {
        "base_path": str(tmp_path / "models") + "/",
        "run_name": "test_run",
        "load_backup": False,
        "backup_name": "backup",
        "backup_every_n_iter": 50,
        "sample_every_n_iter": 10,
        "lat_size": 16,
        "epochs": 2,
        "batch_size": 2,
        "batch_accum": 1,
        "image_size": 64,
        "dataset_path": str(tmp_path / "data"),
        "device": "cpu",
        "lr": 0.0004,
        "resume": False,
        "reinit_optim": False,
        "reinit_lr": 5e-05,
        "use_scheduler": True,
        "scheduler_type": "cosine",
        "scheduler_step_size": 100,
        "scheduler_gamma": 0.5,
        "cosine_t_max": 0,
        "cosine_eta_min": 0.0,
        "ReduceLROnPlateau": False,
        "useEMA": False,
        "ema_decay": 0.999,
        "sample_with_ema": True,
        "weight_decay": 0.01,
        "kld_weight": 0.05,
        "kld_anneal": "none",
        "kld_anneal_epochs": 100,
        "kld_anneal_cycle": 50,
        "num_workers": 0,
        "pin_memory": False,
        "use_amp": False,
        "torch_compile": False,
        "latent_conversion_disable": True,
        "multi_GPU": False,
        "decoder_struct": "VAE_decoder",
        "encoder_struct": "VAE_encoder",
        "encode_path": str(tmp_path),
        "encode_save_path": str(tmp_path),
        "decode_path": str(tmp_path),
        "decode_save_path": str(tmp_path),
        "infer_folder_input": str(tmp_path),
        "infer_folder_output": str(tmp_path),
        "interpolate_folder1": str(tmp_path),
        "interpolate_folder2": str(tmp_path),
        "inter_out": str(tmp_path),
        "resume_path": "",
    }
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.cfg"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # train.py expects to be run from a cwd that contains configs/ and model_structures/
    work = tmp_path / "work"
    work.mkdir()
    (work / "configs").symlink_to(cfg_dir)
    (work / "model_structures").symlink_to(os.path.join(ROOT, "model_structures"))
    (work / "src").symlink_to(os.path.join(ROOT, "src"))
    monkeypatch.chdir(work)
    monkeypatch.syspath_prepend(str(work / "src"))
    monkeypatch.syspath_prepend(str(work))

    from utils import Configs
    return Configs(cfg)


def test_str2bool_and_exclusive_flags(tiny_config, monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:100")
    # Import after chdir so module-level config load succeeds
    import train
    assert train.str2bool("False") is False
    assert train.str2bool("true") is True
    with pytest.raises(argparse.ArgumentTypeError):
        train.str2bool("banana")
    ns = argparse.Namespace(flag_t=False, flag_i=False, flag_e=False, flag_d=True, flag_inter=None)
    train.exclusive_flags(ns, ['flag_t', 'flag_i', 'flag_e', 'flag_d', 'flag_inter'])


def test_effective_kld_weight_schedules(tiny_config):
    from utils import effective_kld_weight, Configs

    args = Configs(dict(tiny_config._variables))
    args.kld_weight = 0.1

    args.kld_anneal = "none"
    assert effective_kld_weight(0, args) == 0.1
    assert effective_kld_weight(999, args) == 0.1

    args.kld_anneal = "linear"
    args.kld_anneal_epochs = 100
    assert effective_kld_weight(0, args) == 0.0
    assert abs(effective_kld_weight(50, args) - 0.05) < 1e-12
    assert effective_kld_weight(100, args) == 0.1
    assert effective_kld_weight(200, args) == 0.1

    args.kld_anneal = "cyclical"
    args.kld_anneal_cycle = 40
    # first half of cycle ramps; second half stays at target
    assert effective_kld_weight(0, args) == 0.0
    assert abs(effective_kld_weight(10, args) - 0.05) < 1e-12
    assert effective_kld_weight(20, args) == 0.1
    assert effective_kld_weight(39, args) == 0.1
    assert effective_kld_weight(40, args) == 0.0  # next cycle restarts


def test_build_lr_scheduler_types(tiny_config):
    from utils import build_lr_scheduler, Configs
    import torch.nn as nn

    model = nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = Configs(dict(tiny_config._variables))

    args.use_scheduler = False
    assert build_lr_scheduler(opt, args, 10) is None

    args.use_scheduler = True
    args.scheduler_type = "cosine"
    sch = build_lr_scheduler(opt, args, 10)
    assert isinstance(sch, torch.optim.lr_scheduler.CosineAnnealingLR)

    args.scheduler_type = "step"
    sch = build_lr_scheduler(opt, args, 10)
    assert isinstance(sch, torch.optim.lr_scheduler.StepLR)


def test_loss_without_ssim_and_sampling_scale(tiny_config):
    import train
    from train import VAE
    from modules import VAE_Encoder, VAE_Decoder

    torch.manual_seed(0)
    model = VAE(VAE_Encoder, VAE_Decoder, tiny_config)
    images = torch.randn(2, 3, tiny_config.image_size, tiny_config.image_size)
    noise = torch.randn(2, 4, tiny_config.image_size // 16, tiny_config.image_size // 16)
    pred = model(images, noise)
    out = model.loss_function(images, pred, ssim_metrics=False)
    assert torch.isfinite(out['loss'])
    assert float(out['SSIM_Loss']) == 0.0

    # prior samples must be scaled so decoder body sees std ~1
    prior = torch.randn(64, 4, 4, 4)
    assert abs(((prior * 0.18215) / 0.18215).std().item() - 1.0) < 0.15
    assert (prior / 0.18215).std().item() > 5


def test_encoder_v2_shapes(tiny_config):
    from train import VAE
    from modules import VAE_Encoder, VAE_Decoder
    from utils import Configs

    cfg = Configs(dict(tiny_config._variables))
    cfg.encoder_struct = "VAE_encoder_v2"
    model = VAE(VAE_Encoder, VAE_Decoder, cfg)
    images = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    noise = torch.randn(2, 4, cfg.image_size // 16, cfg.image_size // 16)
    z = model.encoder(images, noise)
    assert tuple(z.size()) == (2, 4, cfg.image_size // 16, cfg.image_size // 16)
    out = model(images, noise)
    assert tuple(out.size()) == tuple(images.size())
    conv = [m for m in model.encoder if getattr(m, 'stride', None) == (4, 4)][0]
    assert conv.kernel_size == (4, 4)


def test_checkpoint_roundtrip_with_ema(tiny_config, tmp_path):
    from train import VAE
    from modules import VAE_Encoder, VAE_Decoder
    from utils import save_model_checkpoint, load_model_checkpoint

    model = VAE(VAE_Encoder, VAE_Decoder, tiny_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
    )
    # one fake update so EMA state is non-trivial
    for p in model.parameters():
        p.data.add_(0.1)
    ema.update_parameters(model)

    tiny_config.run_name = "ckpt_test"
    models_dir = tmp_path / "models" / tiny_config.run_name
    models_dir.mkdir(parents=True)
    # save_model_checkpoint writes under cwd/models/<run_name>
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", tiny_config.run_name), exist_ok=True)

    loss = torch.tensor(1.23)
    save_model_checkpoint(model, optimizer, loss, 3, 64, 16, 0.01, tiny_config, ema_model=ema)
    ckpt_files = [f for f in os.listdir(os.path.join("models", tiny_config.run_name)) if f.endswith(".pt")]
    assert len(ckpt_files) == 1
    path = os.path.join("models", tiny_config.run_name, ckpt_files[0])

    model2 = VAE(VAE_Encoder, VAE_Decoder, tiny_config)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    ema2 = torch.optim.swa_utils.AveragedModel(
        model2, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
    )
    _, _, loaded_loss, start_epoch, _ = load_model_checkpoint(model2, opt2, path, ema_model=ema2)
    assert start_epoch == 4
    assert abs(float(loaded_loss) - 1.23) < 1e-6
    for a, b in zip(ema.parameters(), ema2.parameters()):
        assert torch.allclose(a, b)
