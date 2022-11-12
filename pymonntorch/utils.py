import torch


def is_number(s):
    """Returns True if string is a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def check_is_torch_tensor(x, device="cpu", dtype=torch.float32):
    if not torch.is_tensor(x):
        try:
            return torch.tensor(x, device=device, dtype=dtype)
        except RuntimeError:
            raise RuntimeError("Could not convert NoneTpe to torch tensor: " + str(x))
