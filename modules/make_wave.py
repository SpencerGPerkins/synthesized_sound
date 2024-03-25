#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:22:27 2024

@author: Spencer Perkins

Module for generating different signals based on sine, square, and saw waves
"""
import numpy as np
import matplotlib.pyplot as plt

class MakeWave:
    
    def __init__(self, duration, sample_rate):
        
        self.dur = duration
        self.sr = sample_rate
        
        self.t = np.linspace(
            0, self.dur, int(self.sr * self.dur), endpoint=False
            )
        
    def sine_wave(self, frequency, gain):
        amp = 10 ** (gain/20)
        sine_wv = amp * np.sin(2 * np.pi * frequency * self.t + 0) # 0 Phase
        return sine_wv
    
    def square_wave(self, frequency, gain):
        amp = 10 ** (gain/20)
        sq_wave = amp * np.sign(np.sin(2 * np.pi * frequency * self.t))
        return sq_wave
    
    def saw_wave(self, frequency, gain):
        amp = 10 ** (gain/20)
        # saw_wv = amp * (np.arctan(np.tan(np.pi * frequency * self.t / 2)) / np.pi)
        saw_wv = amp * (-2/np.pi) * np.arctan(1/np.tan(np.pi * frequency * self.t))
        return saw_wv
    

        

