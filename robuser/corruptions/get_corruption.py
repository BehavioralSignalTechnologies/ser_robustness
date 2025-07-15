from robuser.corruptions.content import ContentCorruption
from robuser.corruptions.gaussian import AWGNAugmentation
from robuser.corruptions.gain_transition import AddGainTransition
from robuser.corruptions.clipping_distortion import AddClippingDistortion
from robuser.corruptions.impulse_response import AddImpulseResponse
from robuser.corruptions.compression import Compression


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
