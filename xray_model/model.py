import torch
import torchxrayvision as xrv
from contextlib import contextmanager


@contextmanager
def allow_full_torch_load():
    """
    Temporarily force torch.load(weights_only=False)
    to support legacy TorchXRayVision checkpoints
    under PyTorch >= 2.6
    """
    original_torch_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_model():
    """
    Load TorchXRayVision DenseNet121 safely
    """
    with allow_full_torch_load():
        model = xrv.models.DenseNet(weights="all")

    model.eval()
    return model


def predict(model, img_tensor):
    with torch.no_grad():
        preds = model(img_tensor)
    return preds
