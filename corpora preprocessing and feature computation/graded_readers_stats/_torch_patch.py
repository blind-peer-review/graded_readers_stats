import torch

# Save original torch.load function
original_torch_load = torch.load

def force_unsafe_torch_load(filename, *args, **kwargs):
    kwargs["weights_only"] = False  # Supress warning
    return original_torch_load(filename, *args, **kwargs)

# Apply the patch globally
torch.load = force_unsafe_torch_load