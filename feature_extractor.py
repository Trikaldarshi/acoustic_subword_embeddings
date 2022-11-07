"""
SSL feature extractor function for various models
Author: Amit Meghanani
Contact: ameghanani1@sheffield.ac.uk

"""
import wave
import torchaudio
import numpy as np
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

sample_rate = 16000
magic_number = sample_rate/hop_length
## freq HuBERT 16000/320 = 50
## 50 Hz is 20 ms --> 20 ms * 16KHz = 320
## sr/hop_length 16000/160 = 100
## 16000/ 512 = 31.25
# Define transform
spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
)

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)



def load_model(model_name,device):
    
    # Select the model:
    if model_name=="HUBERT_BASE":
        bundle = torchaudio.pipelines.HUBERT_BASE

    # Build the model and load pretrained weight.
    model = bundle.get_model().to(device)
    return model,bundle.sample_rate

def SSL_features(path, model, model_sr,layer,device):
    # Load the waveform
    waveform, sample_rate = torchaudio.load(path)

    # Resample audio to the expected sampling rate
    waveform = torchaudio.functional.resample(waveform, sample_rate, model_sr).to(device)

    # Extract acoustic features
    features, _ = model.extract_features(waveform)
    if layer=="all":
        return features
    else:
        return features[layer-1]

def clip_features(feat,st,ed,layer):
    st_v = int(np.floor(50*st))
    ed_v = int(np.ceil(50*ed))
    if layer=="all":
        for i in range(len(feat)):
            feat[i] = feat[i][:,st_v:st_v + ed_v,:]
    else:
        feat = feat[:,st_v:st_v + ed_v,:]
    return feat

def SSL_features_from_wav(waveform, sample_rate, model, model_sr,layer,device):

    # Resample audio to the expected sampling rate
    waveform = torchaudio.functional.resample(waveform, sample_rate, model_sr).to(device)

    # Extract acoustic features
    features, _ = model.extract_features(waveform)
    if layer=="all":
        return features
    else:
        return features[layer-1]

def get_spec(path, device):
    # Load the waveform
    waveform, _ = torchaudio.load(path)
    spec = spectrogram(waveform).to(device)
    return spec.permute(0,2,1)

def get_mfcc(path,device):
    waveform,_ = torchaudio.load(path)
    mfcc = mfcc_transform(waveform).to(device)

    return mfcc.permute(0,2,1)

def clip_features_spec_mfcc(feat,st,ed):
    st_v = int(np.floor(magic_number*st))
    ed_v = int(np.ceil(magic_number*ed))
    print("before clipping", feat.shape)
    feat = feat[:,st_v:st_v + ed_v,:]
    print("after clipping", feat.shape)
    return feat