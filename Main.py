import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment


def mp3_to_wav(filename: str):
    sound = AudioSegment.from_mp3(f'mp3/{filename}.mp3')
    sound.export(f'wav/{filename}.wav', format='wav')


def wav_details(filename: str):
    with wave.open(f'wav/{filename}') as wav:
        print(f'Channels (1 for Mono, 2 for Stereo): {wav.getnchannels()}\n'
              f'Sampling Frequency: {wav.getframerate()}\n'
              f'Audio Frames: {wav.getnframes()}')


"""
Notes:
    
    G: [80000:150000] == 97.019 G2
    D: [1200000:1300000] == 73.206 D2
    A: []
    E: []
"""


def plot_freq_chart(filename: str, show_graphs: bool = True):
    fs, data = wavfile.read(f'wav/{filename}')  # load the data
    data = data[1200000:1300000]  # Control
    data = data / 2.0 ** 15  # Normalize between [-1:1)
    signal = data[:, 0]  # Retrieves the audio data from the pair
    fft_array = np.abs(np.fft.rfft(signal))
    freq = np.fft.rfftfreq(signal.size, d=1. / fs)
    if show_graphs:
        graph_freq(fs, data)
        graph_fft(freq, fft_array)
    print(f'\nFrequency: {freq[np.where(fft_array > 1000)[0][0]]}')


def graph_freq(fs, data):
    length = data.shape[0] / fs
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data[:, 0])
    plt.plot(time, data[:, 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


def graph_fft(freq, fft_array):
    plt.plot(freq[:500], fft_array[:500])
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.show()


if __name__ == '__main__':
    wav_details('StandardBassTuning.wav')
    plot_freq_chart('StandardBassTuning.wav')
