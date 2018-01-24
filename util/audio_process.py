import numpy as np
import librosa

def recovery_phase(amp, n_fft, hop, iters=80):
    phase = 2 * np.pi *np.random.random_sample(amp.shape) - np.pi
    for i in range(iters):
        spec = amp * np.exp(1j * phase)
        raw = librosa.istft(spec, win_length=n_fft, hop_length=hop)
        phase = np.angle(librosa.stft(raw, n_fft=n_fft, hop_length=hop))
    return raw
