#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 16:22:39 2023

@author: Spencer perkins

Wavetable synthesis via wolfsound yt vid
"""

import numpy as np
import scipy.io.wavfile as wav



def linear_interpolation(wave_table, index):
    trunc_index = int(np.floor(index))
    next_index = (trunc_index + 1) % wave_table.shape[0]
    
    next_index_weight = index - trunc_index
    trunc_index_weight = 1 - next_index_weight
    
    return (trunc_index_weight * wave_table[trunc_index] + 
            next_index_weight *
            wave_table[next_index])

def fade_in_out(signal, fade_len = 1000):
    # Multiply signal by amplitude envelope
    # FADE IN : Half cosine window (Hann window)
    # FADE OUT : Flipped half cosine window
    
    fade_in = (1- np.cos(np.linspace(0, np.pi,
                                     fade_len))) * 0.5
    fade_out = np.flip(fade_in)
    
    signal[:fade_len] = np.multiply(fade_in, signal[:fade_len])
    
    signal[-fade_len:] = np.multiply(fade_out, signal[-fade_len:])
    
    return signal

def sawtooth(x):
    return (x + np.pi) / np.pi % 2 - 1



def main():
    sr = 44100
    f = 600 # Frequency set to 440hz
    # f = 220
    t = 10 # Time set to 3 seconds
    waveform = np.sin # Sine waveform
    # waveform = sawtooth
    
    wavetab_len = 64 # Length of wavetable
    wave_table = np.zeros((wavetab_len,)) # Initialize wavetable
    
    for n in range(wavetab_len):
        wave_table[n] =waveform(2 * np.pi * n /
                                wavetab_len)
        
    out = np.zeros((t * sr,))
    
    index = 0 
    
    index_incr = f * wavetab_len / sr # Index increment
    
    for n in range(out.shape[0]):
        # out[n] = wave_table[int(np.floor(index))]
        out[n] =linear_interpolation(wave_table, index)
        index += index_incr
        index %= wavetab_len
    
    gain = -20
    amplitude = 10 ** (gain/20) # Convert to linear amplitude
                                # (inversion of formula for decibels)
    out *= amplitude
    
    out = fade_in_out(out)
    wav.write(
        '../outputs/sine3.wav', sr, out.astype(np.float32))
    
        
    
if __name__ == '__main__':
    main()