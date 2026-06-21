import os
import sys
# Make the repo root importable when run from generator/, and set TSD_ASSETS/TSD_DATA
# env defaults so config ${oc.env:...} interpolation resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import textseedo.paths  # noqa: F401,E402

import glob
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from mgpt.callback import build_callbacks
from mgpt.config import parse_args, instantiate_from_config
from mgpt.data.build_data import build_data
from mgpt.models.build_model import build_model
from mgpt.utils.logger import create_logger
from mgpt.utils.load_checkpoint import load_pretrained, load_pretrained_vae
import pdb

# PyTorch 2.6 changed weights_only default to True. PyTorch Lightning's internal
# checkpoint loading (ckpt_path / RESUME) uses this new default, but checkpoints
# saved with omegaconf objects (e.g. hyperparameters logged by Lightning) contain
# omegaconf types that are not in PyTorch's safe-globals allowlist by default.
# Registering them here allows weights_only=True deserialization to succeed.
try:
    import omegaconf.base
    import omegaconf.nodes
    import omegaconf.listconfig
    import omegaconf.dictconfig
    _omegaconf_safe_globals = [
        omegaconf.base.ContainerMetadata,
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
        omegaconf.nodes.AnyNode,
        omegaconf.nodes.IntegerNode,
        omegaconf.nodes.FloatNode,
        omegaconf.nodes.StringNode,
        omegaconf.nodes.BooleanNode,
        omegaconf.nodes.EnumNode,
    ]
    torch.serialization.add_safe_globals(_omegaconf_safe_globals)
except Exception:
    pass

# PyTorch Lightning's internal checkpoint loading passes weights_only=True by
# default (inherited from PyTorch 2.6 new default), which breaks on checkpoints
# that contain omegaconf objects in their hparams. Patch the low-level loader to
# always use weights_only=False for our trusted checkpoints.
#
# NOTE: torch_io.py does `from cloud_io import _load as pl_load` at import time,
# so patching _cloud_io._load has no effect. We must patch the pl_load *name*
# inside the torch_io module directly.
try:
    import lightning_fabric.plugins.io.torch_io as _torch_io
    _orig_pl_load = _torch_io.pl_load

    def _patched_pl_load(path, map_location=None, **kwargs):
        kwargs["weights_only"] = False
        return _orig_pl_load(path, map_location=map_location, **kwargs)

    _torch_io.pl_load = _patched_pl_load
except Exception:
    pass

def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(OmegaConf.to_yaml(cfg))  # print config file

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tenosrboard' or cfg.LOGGER.WANDB.params.project:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    # ---------------------------------------------------------------------------
    # PRECISION FIX: Use 'bf16-mixed' instead of 'fp16-mixed' or fp32.
    #
    # Problem:  During validation (sanity check), torch.multinomial() crashed
    #           with "probability tensor contains inf, nan or element < 0".
    #           Root cause: fp16 mixed precision has a very narrow dynamic range
    #           (max representable value ≈ 65504). T5 logits — especially early
    #           in training when weights are random — easily overflow that range
    #           to ±inf. torch.multinomial then receives invalid probabilities
    #           and raises a device-side CUDA assertion.
    #
    # Why bf16: bfloat16 has the SAME exponent width as float32 (8 bits), so it
    #           can represent values up to ~3.4e38 and will NOT overflow on
    #           normal neural-network logits. The mantissa is only 7 bits (vs
    #           10 for fp16), so it is less precise numerically, but that is
    #           acceptable for training.
    #
    # GPU requirement: bf16 requires NVIDIA Ampere GPU or newer
    #           (A100, A6000, RTX 3090, RTX 4090, H100, …).
    #           If you move to a pre-Ampere GPU (V100, P100, T4 etc.) you MUST
    #           change this back to '16-mixed' AND apply the autocast(enabled=False)
    #           guard in mgpt_lm.py:generate_direct() to protect generation,
    #           or simply use 32-true (full fp32) at the cost of 2× memory.
    #
    # TODO if changing machines:
    #   Ampere+ (A100, RTX 30/40xx, H100)  →  keep 'bf16-mixed'
    #   Volta / Turing (V100, T4, RTX 20xx) →  use '16-mixed'  (needs fix below)
    #   No AMP support needed               →  use '32-true'
    # ---------------------------------------------------------------------------
    # MEMORY FIX: accumulate_grad_batches=8 with BATCH_SIZE=8 gives an
    #   effective batch size of 64 while using only 1/8th the peak GPU memory
    #   per step compared to the original BATCH_SIZE=64, accumulate=1 setup.
    #   Adjust both values together: effective_bs = BATCH_SIZE * accumulate_grad_batches
    # ---------------------------------------------------------------------------
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        precision='bf16-mixed',
        accumulate_grad_batches=8,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp"
        if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
    )
    logger.info("Trainer initialized")

    # Strict load pretrained model only for non-resume training.
    if cfg.TRAIN.PRETRAINED and not cfg.TRAIN.RESUME:
        load_pretrained(cfg, model, logger)

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # Pytorch 2.0 Compile
    # if torch.__version__ >= "2.0.0":
    #     model = torch.compile(model, mode="reduce-overhead")
    # model = torch.compile(model)

    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.RESUME)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
