#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:58:51 2024

@author : spencer perkins

Module : Basic utilities, feature extraction, noise generator
"""
import numpy as np 
import librosa as l 
from scipy.signal import lfilter
import librosa.display as display
import matplotlib.pyplot as plt

class BasicUtils:
    
    def load(audio, samp_rate):
        """ Load Audio
        Params
        ------
        audio : string, path two audio file
        samp_rate : sampling rate
        
        returns
        -------
        (sig, sr) : tuple
        sig :array of shape ((duration*sampling rate), ) 
        sr : int
        """
        
        sig, sr = l.load(audio, sr=samp_rate, mono=True)
        
        return (sig, sr)
    
    def pad(sig, pad_len):
        """ Add Padding to array
        Params
        ---------
        sig : array, original signal
        pad_len : int, how many time steps to add
        
        Returns
        -------
        padding : array of length len(sig) + pad_len
        """
        
        pad = np.zeros((len(sig), pad_len))
        padding = np.concatenate((sig, pad), axis=1)

        return padding
        
    
class Features:
    
    def __init__(self, sr, n_fft, hop_len, win_len):
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        
    
    def stft(self, sig):
        """ Short-time Fourier Transform
        Params
        --------
        sig : array, input 1D signal
        n_fft : int, length of fft window
        hop_len : int, if None, equal to n_fft//2
        win_len : int, window length
        
        Returns
        ---------
        S : 2D time-frequency representation of the 1D signal 
        shape (1+n_fft//2, (samp_rate*duration)/(n_fft-hop_len)) 
        """
        
        if self.hop_len == None:
            hop_len = self.n_fft//2
        else:
            hop_len = self.hop_len
        
        S = l.stft(
            y=sig, n_fft=self.n_fft,
                   hop_length=hop_len, win_length=self.win_len
                   )
        print(f'S {S.shape}')
        return(np.abs(S))
    
    def mel_spec(self, sig, mels: int=40, power: int=2, 
                 fmin: float=0., fmax: float=None,
                 log: bool=False):
        
        if self.hop_len == None:
            hop_len = self.n_fft//2
        else:
            hop_len = self.hop_len
            
        if fmax == None:
            fmax = self.sr / 2.0
        
        S = l.feature.melspectrogram(
            y=sig, sr=self.sr, 
            n_fft=self.n_fft, win_length=self.win_len,
            hop_length=hop_len, n_mels=mels,
            power=power, fmin=fmin, fmax=fmax
            )
        
        if log == True:
            S = l.amplitude_to_db(S, ref=np.max)
        
        return S
    
    def pcen(sig, alpha: float=0.8,
             delta: float=10., r: float=0.25,
             s: float=0.00025):
        
        eps = 10e-6

        M = [sig[..., 0]]
        for i in range(1, sig.shape[-1]):
            M.append((1-s) * M[-1] + s * sig[..., i])
        
        M = np.stack(M, -1)
        agc = sig / (eps + M)**alpha
        drc = (agc + delta)**r - delta**r
        return drc
        
    #def bandpass():

class NoiseGen:
    
    def white(sig, std: float=1):
        
        dur = len(sig)
        wh_noise = np.random.normal(0, std, dur)
        
        return sig + wh_noise
    
    def brown(sig):
        
        dur = len(sig)
        # Generate white noise
        white_noise = np.random.normal(0, 1, dur)

        # Create a one-pole low-pass filter (integrator)
        b = [1]  # Numerator coefficients
        a = [1, -0.99]  # Denominator coefficients (for a first-order IIR filter)
        filtered_noise = lfilter(b, a, white_noise)

        # Scale the filtered noise to maintain its variance
        filtered_noise *= np.sqrt(np.var(white_noise) / np.var(filtered_noise))
    
        return sig + filtered_noise
    



        