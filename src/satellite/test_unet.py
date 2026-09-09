import torch

from unet import UNet


print("=" * 60)
print("        OILTRACE-AI - U-NET TEST")
print("=" * 60)


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------

model = UNet(
    in_channels=1,
    out_channels=1
)


print("\nModel created successfully.")


# ---------------------------------------------------------
# Create dummy batch
# ---------------------------------------------------------

x = torch.randn(
    4,
    1,
    256,
    256
)


print("\nInput")
print("-" * 60)
print("Shape :", x.shape)


# ---------------------------------------------------------
# Forward pass
# ---------------------------------------------------------

with torch.no_grad():
    output = model(x)


print("\nOutput")
print("-" * 60)
print("Shape :", output.shape)
print("Dtype :", output.dtype)


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert output.shape == (
    4,
    1,
    256,
    256
)

assert output.dtype == torch.float32


print("\nForward pass PASSED!")
print("Input and output spatial dimensions MATCH.")


# ---------------------------------------------------------
# Parameter count
# ---------------------------------------------------------

parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nTrainable parameters :", f"{parameters:,}")


print("\n" + "=" * 60)
print("U-Net test completed.")
print("=" * 60)