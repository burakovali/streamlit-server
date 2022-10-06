# Imports
import datetime
import argparse
from re import S
import sys

# from more_itertools import sliding_window
from pydub import AudioSegment
import numpy as np
import scipy
from scipy.io import wavfile
from scipy.io.wavfile import write
from scipy.signal import find_peaks
from scipy import interpolate

# from plotly.offline import init_notebook_mode
# import plotly.graph_objs as go
# import plotly
# import matplotlib.pyplot as plt

from scipy.spatial import distance

# import IPython.display as ipd
import librosa
import librosa.display
import random
import math

# from sklearn.preprocessing import scale
import makelab
from makelab import audio
from makelab import signal
import itertools
from scipy.signal import butter, filtfilt

# import sklearn
# from sklearn.metrics import mean_squared_error
import scipy.stats
from scipy.spatial.distance import directed_hausdorff

# import similaritymeasures
import getpass
import time
import os
from datetime import datetime

########################


def pad_zeros_right(s, padding_length):
    # https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    return np.pad(s, (0, padding_length), mode="constant", constant_values=0)


def pad_mean_right(s, padding_length):
    # https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    return np.pad(s, (0, padding_length), mode="mean")


# Some functions to use
def read_audio(filename):
    fs_wav, data_wav = wavfile.read(filename)
    time_wav = np.arange(0, len(data_wav)) / fs_wav
    print(
        "filename={} \nduration={} secs \nSampling rate={} Hz".format(
            filename, data_wav.shape[0] / fs_wav, fs_wav
        )
    )
    return data_wav, time_wav, fs_wav


def normalize(input_array, sampling_bits):
    normalized_array = input_array / (2 ** (sampling_bits - 1))
    # normalized_array = librosa.util.normalize(input_array)
    return normalized_array


def crop(input_array, duration, sampling_period):
    cropped = input_array[0 : duration * sampling_period]
    return cropped


def crop_from_left(input_array, idx, sampling_period):
    cropped = input_array[idx : (len(input_array) - 1) * sampling_period]
    return cropped


def crop_from_right(input_array, duration, sampling_period):
    cropped = input_array[0 : (len(input_array) - 1 - duration) * sampling_period]
    return cropped


def draw(horizantal, vertical, label):
    plotly.offline.iplot({"data": [go.Scatter(x=vertical, y=horizantal, name=label)]})


def first_peak(input_array, prominence):
    peaks, _ = find_peaks(input_array, prominence=prominence)
    first_peak = peaks[0]
    # plt.subplot(2,2,2)
    # plt.plot(peaks, input_array[peaks],"ob")
    # plt.plot(input_array)
    # plt.legend(['prominence'])
    # largest_val = abs(max(input_array,key=abs))
    return first_peak


def scale_to_unity(X):
    # Let us normalize to unity by min-max
    X_manual_scaled = (X - X.min()) / (X.max() - X.min())
    return X_manual_scaled


def break_to_segments(signal, duration, sampling_period):
    signal_len = len(signal)
    segment_size_t = duration  # seconds
    segment_size = int(segment_size_t * sampling_period)
    # Break signal into list of segments in a single-line Python code
    segments = np.array(
        [signal[x : x + segment_size] for x in np.arange(0, signal_len, segment_size)]
    )
    return segments


def locate_last_sync_pulse(
    input_signal, numof_sync_pulses, sync_pulse_duration, input_sr
):

    # Normalization is performed for the dynamic range
    # which is determined by the number of bits used during
    # quantization
    # input_signal = normalize(input_signal,quantization_bits)
    # limit
    # in_signal = input_signal
    # input_signal = crop(input_signal,60,input_sr) # limit to duration secs
    # Scaling to unity is just a shift of amplituedes from [-1 - 1] to [0 - 1] range
    # input_signal = scale_to_unity(input_signal)
    # signal.plot_audio(input_signal,input_sr)
    duration_tolerance = (sync_pulse_duration * 40) / 100
    non_silent_interval = librosa.effects.split(input_signal, top_db=30)
    # print(non_silent_interval)

    match = 0
    for _x in non_silent_interval:
        print("Between: ", _x[0] / input_sr, "--", _x[1] / input_sr, "secs", end=" ")
        print("duration = ", (_x[1] - _x[0]) / input_sr, "secs", end=" ")
        range_end = _x[1]
        duration = (_x[1] - _x[0]) / input_sr

        # First check if pulse is similar to what we are looking for in duration
        if ((sync_pulse_duration - duration_tolerance) <= duration) and (
            (sync_pulse_duration + duration_tolerance) >= duration
        ):
            match += 1
            # make sure that located pulse is an element of train
            # otherwise discard all previous pulses and assume
            # that this one is the start of pulse train
            if match > 1:
                silence_duration = (_x[0] - crop_offset) / input_sr
                # print("silence duration = ",silence_duration, end=" ")
                if not ((sync_pulse_duration + duration_tolerance) >= silence_duration):
                    print("\t***new start***", end="")
                    match = 1

            print("\tMatched number= ", match)
            if numof_sync_pulses == match:
                crop_offset = _x[1]  # no more cropping from the last pulse
                print(
                    "Synchronization pulse found cropping from left idx=", crop_offset
                )
                # Crop reference signal from left
                input_signal = crop_from_left(input_signal, crop_offset, input_sr)
                # Further tuning to get rid of silence at the begining would improve
                # alignment of the signal with others
                # print("Searching for first peak...")
                # tmp_signal = crop(input_signal, 5, input_sr)
                # peak_idx = first_peak(tmp_signal, 0.2)
                # peak_idx = first_peak(input_signal, 0.01)
                # print("Located first peak, cropping from left idx=", peak_idx)
                # input_signal = crop_from_left(input_signal, peak_idx, input_sr)
                print("Start of signal aligned to last sync pulse.")
                # return input_signal, (crop_offset + peak_idx)
                return input_signal, crop_offset
            # Store the idx of (latest-1)th  pulse in memory so that when
            # we are done with locating, we crop signal from the stored pulse
            # which later solves a lot of trouble while locating the very first
            # peak in the signal
            crop_offset = _x[1]

        else:
            print("Segment does not match!")

    return input_signal, 0


def compute_visqol_correlation(
    input_signal,
    input_sr,
    test_signal,
    test_sr,
    fidelity_threshold=0.5,
    wduration=1,
    sampling_bits=16,
    save=False,
    smoothing="past&future",
):

    valid_filters = ["past", "past&future"]
    if smoothing not in valid_filters:
        raise ValueError("Unknown smoothing type!")
    unscaled_input_signal = input_signal
    unscaled_test_signal = test_signal
    # input_signal =  librosa.util.normalize(input_signal)
    # input_signal = scale_to_unity(input_signal)
    # test_signal = librosa.util.normalize(test_signal)
    # test_signal = scale_to_unity(test_signal)

    # Before comparing signals make sure over all length of the signal arrays are same
    if len(input_signal) != len(test_signal):
        if len(input_signal) > len(test_signal):
            input_signal = input_signal[0 : len(test_signal)]
        else:
            test_signal = test_signal[0 : len(input_signal)]

    # Now let us make sure that the overall duration of the files are multiples
    # of 10 seconds which we generate MOS scores for.
    frame_duration = wduration
    frame_size = test_sr * frame_duration
    duration_of_the_signal = int(len(test_signal) / test_sr)  #
    duration_of_the_signal -= (
        duration_of_the_signal % frame_duration
    )  # find remainder and subtract it from total duration
    test_signal = test_signal[0 : duration_of_the_signal * test_sr]
    input_signal = input_signal[0 : duration_of_the_signal * input_sr]

    # at that point we made sure that audio file sizes are same and multiples of 10 seconds
    # and synchronization pulses are stripped out of audio files

    # Now, we will break them into 10 seconds frames for computing MOS scores for each of them
    ref_frames = break_to_segments(input_signal, frame_duration, input_sr)
    test_frames = break_to_segments(test_signal, frame_duration, test_sr)
    frame_no = 0
    t = 0
    sr = input_sr
    i = 0
    result_list = []
    mos = 4
    import subprocess

    for (s1, s2) in zip(ref_frames, test_frames):
        # Save 10 seconds of frame in a file
        print(f"frame no={frame_no}, frame[{t}-{t+frame_duration} secs]: ", end=" ")
        wavfile.write(f"ref-visqol.wav", input_sr, s1.astype(np.int16))
        wavfile.write(f"test-visqol.wav", test_sr, s2.astype(np.int16))
        # and now pass them over to visqol for computing the MOS score

        p = subprocess.Popen(
            # "ls visqol -lat",
            "./bazel-bin/visqol --reference_file ../ref-visqol.wav --degraded_file ../test-visqol.wav --verbose",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd="visqol",
        )

        for line in p.stdout.readlines():
            pos = line.find(b"MOS-LQO:")
            if pos != -1:
                line = line.decode("UTF-8")
                tokens = line.split()
                mos = tokens[1]
                print(f"mos = {mos}")

            retval = p.wait()

        result_list.append([t, t + frame_duration, mos])

        t += frame_duration
        i = i + 1
        frame_no += 1

    return result_list, len(test_signal)


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs * 4
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype="lowpass", analog=False)
    y = filtfilt(b, a, data)
    return y


def match_align(pulse_repeat, pulse_duration, filename):
    # first tunings
    crop_offset = 0
    max_duration = 90
    quantization_bits = 16
    # NEW_SAMPLERATE = 44000

    # first read audio signals from file
    input_signal, arg_time_array, input_sr = read_audio(filename)  # read from file

    input_signal = audio.convert_to_mono(
        input_signal
    )  # incase it is stero convert it to mono

    input_signal, input_cropped = locate_last_sync_pulse(
        input_signal=input_signal,
        input_sr=input_sr,
        numof_sync_pulses=pulse_repeat,
        sync_pulse_duration=pulse_duration,
    )

    return input_signal
