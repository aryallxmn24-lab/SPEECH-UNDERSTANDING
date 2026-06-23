import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    num_frames = 1 + (len(waveform) - frame_length) // step
    frames = np.array([
        waveform[i * step:i * step + frame_length]
        for i in range(num_frames)
    ])
    return frames

def frames_to_mstft(frames):
    mstft = np.abs(np.fft.fft(frames, axis=1))
    return mstft

def mstft_to_spectrogram(mstft):
    floor = 0.001 * np.amax(mstft)
    spectrogram = 20 * np.log10(np.maximum(mstft, floor))
    return spectrogram