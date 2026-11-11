"""Training-time augmentation: random changes that could happen in a real chest X-ray
(Step 1.2). Never used for val or test."""

from torchvision.transforms import v2


def build_train_transform(aug):
    """Random augmentation for one training image, built from the config's `augmentation`
    section (a dict). aug None -> returns None (no augmentation).
    Input and output: image tensor (1, size, size), float in 0..1."""
    if aug is None:
        return None
    steps = []
    if aug["horizontal_flip"]:
        steps.append(v2.RandomHorizontalFlip(p=0.5))
    # colour before the affine, so the black corners it fills stay black
    steps.append(v2.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]))
    steps.append(
        v2.RandomAffine(
            degrees=aug["rotation_degrees"],
            translate=(
                aug["translate"],
                aug["translate"],
            ),
            # (horizontal, vertical): the same fraction for both
            scale=tuple(aug["scale"]),
            interpolation=v2.InterpolationMode.BILINEAR,  # the default, nearest, makes blocky edges
        )
    )
    return v2.Compose(steps)
