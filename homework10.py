import numpy as np
import torch
import torch.nn as nn

def get_features(waveform, Fs):
    preemph = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)

    num_frames = 1 + (len(preemph) - frame_length) // step

    frames = np.array([
        preemph[i*step:i*step+frame_length]
        for i in range(num_frames)
    ])

    features = np.abs(np.fft.fft(frames, axis=1))
    features = features[:, :frame_length // 2]

    vad_length = int(0.025 * Fs)
    vad_step = int(0.010 * Fs)

    vad_frames = 1 + (len(waveform) - vad_length) // vad_step

    energy = np.array([
        np.sum(waveform[i*vad_step:i*vad_step+vad_length]**2)
        for i in range(vad_frames)
    ])

    threshold = 0.1 * np.max(energy)

    labels = np.zeros(num_frames, dtype=int)

    segment_id = 1
    for i in range(vad_frames):
        if energy[i] > threshold:
            start_sample = i * vad_step
            end_sample = start_sample + vad_length

            start_frame = start_sample // step
            end_frame = min(num_frames, end_sample // step)

            labels[start_frame:end_frame] = segment_id
            segment_id += 1

    labels = labels[:num_frames]

    return features, labels

def train_neuralnet(features, labels, iterations):
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    nfeats = features.shape[1]
    nlabels = int(np.max(labels)) + 1

    model = nn.Sequential(
        nn.LayerNorm(nfeats),
        nn.Linear(nfeats, nlabels)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    lossvalues = np.zeros(iterations)

    for i in range(iterations):
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        lossvalues[i] = loss.item()

    return model, lossvalues

def test_neuralnet(model, features):
    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1).detach().numpy()

    return probabilities