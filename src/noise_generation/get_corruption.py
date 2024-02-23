from noise_generation.content import ContentCorruption
from noise_generation.gaussian import AWGNAugmentation


def get_corruption(corruption_name):
    if corruption_name == "content":
        return ContentCorruption
    elif corruption_name == "gaussian":
        return AWGNAugmentation
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
