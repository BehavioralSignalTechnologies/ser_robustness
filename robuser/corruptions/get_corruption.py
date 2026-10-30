from robuser.corruptions.content import ContentCorruption
from robuser.corruptions.gaussian import AWGNAugmentation
from robuser.corruptions.gain_transition import AddGainTransition
from robuser.corruptions.clipping_distortion import AddClippingDistortion
from robuser.corruptions.impulse_response import AddImpulseResponse
from robuser.corruptions.compression import Compression
from robuser.corruptions.noise_snr import NoiseSNR
from robuser.corruptions.rir import RIRConvolution
from robuser.corruptions.clipping_sdr import ClippingSDR
from robuser.corruptions.level import Level
from robuser.corruptions.telephony_codec import TelephonyCodec
from robuser.corruptions.chain import Chain


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
    # Labelled corruptions: return the exact applied level (SNR, C50, SDR) via run_with_info
    elif corruption_name == "noise_snr":
        return NoiseSNR
    elif corruption_name == "rir":
        return RIRConvolution
    elif corruption_name == "clipping_sdr":
        return ClippingSDR
    elif corruption_name == "level":
        return Level
    elif corruption_name == "telephony_codec":
        return TelephonyCodec
    elif corruption_name == "chain":
        return Chain
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
