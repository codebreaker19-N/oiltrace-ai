import torch

from losses import DiceLoss, BCEDiceLoss


print("=" * 60)
print("        OILTRACE-AI - LOSS FUNCTION TEST")
print("=" * 60)


# ---------------------------------------------------------
# Dummy prediction and target
# ---------------------------------------------------------

logits = torch.randn(
    4, 1, 256, 256
)

targets = torch.randint(
    0,
    2,
    (4, 1, 256, 256)
).float()


# ---------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------

dice_loss = DiceLoss()

dice_value = dice_loss(
    logits,
    targets
)


print("\nDice Loss")
print("-" * 60)
print("Value :", dice_value.item())


# ---------------------------------------------------------
# BCE + Dice
# ---------------------------------------------------------

combined_loss = BCEDiceLoss()

total_value = combined_loss(
    logits,
    targets
)


print("\nBCE + Dice Loss")
print("-" * 60)
print("Value :", total_value.item())


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

assert torch.isfinite(dice_value)
assert torch.isfinite(total_value)

assert dice_value.item() >= 0
assert total_value.item() >= 0


print("\nAll loss checks PASSED!")

print("\n" + "=" * 60)