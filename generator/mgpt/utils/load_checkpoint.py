import torch
import pdb
def load_pretrained(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    state_dict = torch.load(ckpt_path, map_location="cpu",weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model


def load_pretrained_vae(cfg, model, logger=None):
    """Load pretrained motion tokenizer checkpoint.

    Delegates to model.vae.load_pretrained() so the FSQTokenizer
    wrapper handles checkpoint-format details.
    """
    vae_path = cfg.TRAIN.PRETRAINED_VAE
    if logger is not None:
        logger.info(f"Loading pretrained VAE from {vae_path}")

    vae = getattr(model, "vae", None) or getattr(model, "motion_vae", None)
    if vae is None:
        raise AttributeError("Model has neither 'vae' nor 'motion_vae' attribute")

    # Prefer the unified load_pretrained() API (MotionTokenizerBase)
    if hasattr(vae, "load_pretrained"):
        vae.load_pretrained(vae_path)
    else:
        # Fallback for raw models without the wrapper
        checkpoint = torch.load(vae_path, map_location="cpu", weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        vae.load_state_dict(state_dict, strict=True)

    if logger is not None:
        logger.info("Successfully loaded VAE weights")
    return model