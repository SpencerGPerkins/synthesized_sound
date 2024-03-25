#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:29:40 2024

@author: Spencer Perkins

Module : Padding front, back
"""
import numpy as np

class Padding:
    
    def __init__(self, sample_rate, seconds):
        
        self.samples = sample_rate * seconds
    
    def pad_front(self, signal):
        
        pad_len = self.samples - len(signal)
        pad = np.zeros(int(pad_len))
        print(pad.shape)
        padded_front = np.concatenate((pad, signal))
        
        return padded_front
    
    def pad_back(self, signal):
        
        pad_len = self.samples - len(signal)
        pad = np.zeros(int(pad_len))
        padded_back = np.concatenate((signal, pad))
        
        return padded_back
    
    def pad_both(self, signal):
        
        pad_len = (self.samples - len(signal)) / 2
        pad = np.zeros(int(pad_len))
        padded = np.concatenate((pad, signal, pad))
        
        return padded


            
                
        
     
        
