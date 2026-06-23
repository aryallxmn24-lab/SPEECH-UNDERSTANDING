import numpy as np

def VAD(waveform, Fs):
    frame_length = int(0.025 * Fs)
    step = int(0.010 * Fs)

    num_frames = 1 + (len(waveform) - frame_length) // step

    energies = np.array([
        np.sum(waveform[i*step:i*step+frame_length]**2)
        for i in range(num_frames)
    ])

    threshold = 0.1 * np.max(energies)

    segments = []
    current = []

    for i in range(num_frames):
        start = i * step
        end = start + frame_length

        if energies[i] > threshold:
            current.extend(waveform[start:end])
        elif len(current) > 0:
            segments.append(np.array(current))
            current = []

    if len(current) > 0:
        segments.append(np.array(current))

    return segments

def segments_to_models(segments, Fs):
    models = []

    for segment in segments:
        preemph = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])

        frame_length = int(0.004 * Fs)
        step = int(0.002 * Fs)

        num_frames = 1 + (len(preemph) - frame_length) // step

        frames = np.array([
            preemph[i*step:i*step+frame_length]
            for i in range(num_frames)
        ])

        mstft = np.abs(np.fft.fft(frames, axis=1))
        mstft = mstft[:, :frame_length//2]

        floor = 0.001 * np.max(mstft)
        spectrogram = 20 * np.log10(np.maximum(mstft, floor))

        model = np.mean(spectrogram, axis=0)
        models.append(model)

    return models

def recognize_speech(testspeech, Fs, models, labels):
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    sims = np.zeros((len(models), len(test_models)))
    test_outputs = []

    for k, test_model in enumerate(test_models):
        for y, model in enumerate(models):
            sims[y, k] = np.dot(model, test_model) / (
                np.linalg.norm(model) * np.linalg.norm(test_model)
            )

        best = np.argmax(sims[:, k])
        test_outputs.append(labels[best])

    return sims, test_outputs