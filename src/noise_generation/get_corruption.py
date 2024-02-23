from noise_generation.content import ContentCorruption


def get_corruption(corruption_name):
    if corruption_name == "content":
        return ContentCorruption
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")

