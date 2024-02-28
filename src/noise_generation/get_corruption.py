from noise_generation.content import ContentCorruption
from noise_generation.gaussian import AWGNAugmentation
from noise_generation.gain_transition import AddGainTransition
from noise_generation.clipping_distortion import AddClippingDistortion
from noise_generation.impulse_response import AddImpulseResponse
from noise_generation.compression import Compression


def get_corruption(corruption_name):
    if corruption_name == "content":
        return ContentCorruption
    elif corruption_name == "gaussian":
        return AWGNAugmentation
    elif corruption_name == "gain_transition":
        return AddGainTransition
    elif corruption_name == "clipping_distortion":
        return AddClippingDistortion
    elif corruption_name == "impulse_response":
        return AddImpulseResponse   
    elif corruption_name == "compression":
        return Compression
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
