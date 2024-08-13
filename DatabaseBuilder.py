
from scipy.io.wavfile import read
import librosa
import numpy as np
from pydub import AudioSegment
import os 
import glob
from tqdm import tqdm
from typing import List, Dict, Tuple
from scipy.signal import find_peaks
import pickle
from scipy import signal
from scipy.signal import find_peaks, peak_prominences

def fingerprintMap5(audio, Fs):

    audio, Fs = librosa.load(audio)
    # Parameters
    window_length_seconds = 2
    window_length_samples = int(window_length_seconds * Fs)
    window_length_samples += window_length_samples % 2
    num_peaks = 6
    # Pad the song to divide evenly into windows
    amount_to_pad = window_length_samples - audio.size % window_length_samples
    song_input = np.pad(audio, (0, amount_to_pad))
    # Perform a short time fourier transform
    frequencies, times, stft = signal.stft(
        song_input, Fs, nperseg=window_length_samples, nfft=window_length_samples, return_onesided=True
    )#function returns the frequencies, times, and STFT coefficients of the audio data.
    constellation_map = []
    for time_idx, window in enumerate(stft.T):
        # Spectrum is by default complex. 
        # We want real values only
        spectrum = abs(window)
        # Find peaks - these correspond to interesting features
        # Note the distance - want an even spread across the spectrum
        peaks, props = signal.find_peaks(spectrum, prominence=0.0001, distance=200)
        # Only want the most prominent peaks
        # With a maximum of 15 per time slice
        n_peaks = min(num_peaks, len(peaks))
        # Get the n_peaks largest peaks from the prominences
        # This is an argpartition
        # Useful explanation: https://kanoki.org/2020/01/14/find-k-smallest-and-largest-values-and-its-indices-in-a-numpy-array/
        largest_peaks = np.argpartition(props["prominences"], -n_peaks)[-n_peaks:]
        for peak in peaks[largest_peaks]:
            frequency = frequencies[peak]
            constellation_map.append([time_idx, frequency])
    return constellation_map







def create_hashes5(constellation_map, song_id=None):
    '''Time resolution: If the time difference between two pairs is too small (1 or less), 
    it suggests that the two pairs occur very close together in time. At such a small time difference, 
    the frequency values are more likely to be within the same audio event or waveform cycle. 
    Including such very small time differences may lead to capturing more noise than actual distinguishable patterns.
Time range: If the time difference between two pairs is too large (greater than 10), 
it indicates that the two pairs are far apart in time. 
Including such large time differences might introduce more variations and make it harder to find meaningful patterns
 or similarities between the frequencies.
    '''
    hashes = {}
    # Use this for binning - 23_000 is slighlty higher than the maximum
    # frequency that can be stored in the .wav files, 22.05 kHz
    upper_frequency = 23_000 
    frequency_bits = 10
    # Iterate the constellation
    for idx, (time, freq) in enumerate(constellation_map):
        # Iterate the next 100 pairs to produce the combinatorial hashes
        # When we produced the constellation before, it was sorted by time already
        # So this finds the next n points in time (though they might occur at the same time)
        for other_time, other_freq in constellation_map[idx : idx + 500]: 
            diff = other_time - time
            # If the time difference between the pairs is too small or large
            # ignore this set of pairs
            if diff <= 1 or diff > 10:
                continue
            # Place the frequencies (in Hz) into a 1024 bins
            freq_binned = freq / upper_frequency * (2 ** frequency_bits)
            other_freq_binned = other_freq / upper_frequency * (2 ** frequency_bits)
            # Produce a 32 bit hash
            # Use bit shifting to move the bits to the correct location
            hash = int(freq_binned) | (int(other_freq_binned) << 10) | (int(diff) << 20)
            hashes[hash] = (time, song_id)
    return hashes

def main():
 
    
    
    song_name_index = {}
    database: Dict[int, List[Tuple[int, int]]] = {}
    
    
    # Go through each song, using where they are alphabetically as an id
    folder_path = r"C:\Users\samar\OneDrive\Desktop\GradProject2\RecogSongs"
    songs = glob.glob(folder_path + '\*.mp3')  # Change the extension to '.mp3    
    wav_file = 'output.wav'
   
    for index, filename in enumerate(tqdm(sorted(songs))):
       basename = os.path.basename(filename)
       basename_without_zeros = basename.lstrip('0').split('.')[0]
       song_name_index[index] =  basename_without_zeros
    # Read the song, create a constellation and hashes
       print(filename)
       #audio = AudioSegment.from_mp3(filename)
       #audio.export(wav_file, format='wav')
    
       constellation = fingerprintMap5(filename,1)
       hashes = create_hashes5(constellation, index)

    # For each hash, append it to the list for this hash
       for hash, time_index_pair in hashes.items():
         if hash not in database:
            database[hash] = []
         database[hash].append(time_index_pair)
    # Dump the database and list of songs as pickles
    with open("database1.pickle", 'wb') as db:
     pickle.dump(database, db, pickle.HIGHEST_PROTOCOL)
    with open("song_index1.pickle", 'wb') as songs:
     pickle.dump(song_name_index, songs, pickle.HIGHEST_PROTOCOL)


main()