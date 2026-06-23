import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    nframes = 1 + (len(speech) - frame_length) // frame_skip

    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        a = librosa.lpc(frame, order=order)
        A[i, :] = a

        e = np.convolve(frame, a, mode='same')
        excitation[i, :] = e

    return A, excitation

def synthesize(e, A, frame_skip):
    nframes = A.shape[0]
    order = A.shape[1] - 1

    duration = nframes * frame_skip
    synthesis = np.zeros(duration)

    for i in range(nframes):
        start = i * frame_skip
        stop = min(start + frame_skip, len(e))

        for n in range(start, stop):
            y = e[n]

            for k in range(1, order + 1):
                if n - k >= 0:
                    y -= A[i, k] * synthesis[n - k]

            synthesis[n] = y

    return synthesis

def robot_voice(excitation, T0, frame_skip):
    nframes = excitation.shape[0]

    gain = np.sqrt(np.mean(excitation**2, axis=1))

    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):
        start = i * frame_skip
        end = start + frame_skip

        for n in range(start, end, T0):
            e_robot[n] = gain[i]

    return gain, e_robot