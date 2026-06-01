import numpy as np

def major_chord(f, Fs):
    
    N = int(Fs / 2)  # half second duration
    n = np.arange(N)

    # Root, major third (+4 semitones), major fifth (+7 semitones)
    f_root = f
    f_third = f * 2**(4/12)
    f_fifth = f * 2**(7/12)

    x = (np.cos(2*np.pi*f_root*n/Fs) +
         np.cos(2*np.pi*f_third*n/Fs) +
         np.cos(2*np.pi*f_fifth*n/Fs))

    return x


def dft_matrix(N):
   
    n = np.arange(N)
    k = n.reshape((N, 1))

    W = np.cos(2*np.pi*k*n/N) - 1j*np.sin(2*np.pi*k*n/N)

    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    N = len(x)

    # DFT
    W = dft_matrix(N)
    X = np.dot(W, x)

    # Magnitude spectrum
    mag = np.abs(X)

    # Only positive frequencies
    mag = mag[:N//2]

    # Find indices of three largest peaks
    idx = np.argsort(mag)[-3:]

    # Convert indices to frequencies
    freqs = idx * Fs / N

    # Sort frequencies
    freqs = np.sort(freqs)

    f1, f2, f3 = freqs

    return f1, f2, f3