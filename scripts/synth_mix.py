#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:57:12 2024

@author: Spencer Perkins

Script : Synthetic Signal Mixtures
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import librosa.display as display
import scipy.io.wavfile as wav

import sys
sys.path.insert(1, '../modules/')
from make_wave import MakeWave
from utils import Features, NoiseGen
from padding import Padding

class ExtraUtils:
    
    def __init__(self, sr, duration):
        
        self.samprate = sr
        self.dur = duration
        
    def synth_gen(self):
        
        waves = {'sine':[], 'square':[], 'saw':[]} # Independent sound waves
        
        for i in range(1000):
            # Initialize wave generator with random duration
            wave_generator = MakeWave(
                duration=np.random.uniform(2., 8.), sample_rate=self.samprate
                ) 
            sine_wave = wave_generator.sine_wave(
                np.random.uniform(0., self.samprate/2), np.random.uniform(-60., 1.)
                )
            sine_wave = self.pad_signal( sine_wave)
            waves['sine'].append(sine_wave)
            
            # Re-initialize wave generator for new duration
            wave_generator = MakeWave(
                duration=np.random.uniform(2., 8.), sample_rate=self.samprate
                )
            square_wave = wave_generator.square_wave(
                np.random.uniform(0., self.samprate/2), np.random.uniform(-60., 1.)
                )
            square_wave = self.pad_signal(square_wave)
            waves['square'].append(square_wave)
            
            # Re-initialize wave generator for new duration
            wave_generator = MakeWave(
                duration=np.random.uniform(2., 8.), sample_rate=self.samprate
                )
            saw_wave = wave_generator.saw_wave(
                np.random.uniform(0., self.samprate/2), np.random.uniform(-60., 1.)
                )
            saw_wave = self.pad_signal(saw_wave)
            waves['saw'].append(saw_wave)
            
            if i % 10 ==0:
                print(f'{i}th wave generated')
        
        return waves
    
    def pad_signal(self, wave, back=True):
        
        Pad = Padding(self.samprate, self.dur)
    
        if ((self.samprate * self.dur) - len(wave)) % 2 == 0:
            n_wave = Pad.pad_both(wave)   
        elif back == True:
            n_wave = Pad.pad_back(wave)
        elif back == False:
            n_wave = Pad.pad_front(wave)
            
        return n_wave

class SaveAudioFile:
    
    def __init__(self, sr):
        self.sr = int(sr)
        
    def wav_file(self, path, waves):
        wav_path = path
        for i in range(20):
            wav.write(
                wav_path+'sine/'+str(i+1)+'.wav',
                self.sr, waves['sine'][i].astype(np.float32)
                )
            wav.write(
                wav_path+'square/'+str(i+1)+'.wav',
                self.sr, waves['square'][i].astype(np.float32)
                )
            wav.write(
                wav_path+'saw/'+str(i+1)+'.wav',
                self.sr, waves['saw'][i].astype(np.float32)
                )
            
        print('\nWav files saved!\n')
        
    def mix_file(self, path, mixtures):
        for m, mix in enumerate(mixtures):
            wav.write(
                path+str(m+1)+'.wav',
                self.sr, mix.astype(np.float32)
                )
            print(f'Mixture file : {m + 1}')
        print('\nMix files Saved!\n')
        
class SpecVis:
    
    def __init__(self, sr, hop_size):
        
        self.sr = sr
        self.hop_size = hop_size
        
    def vis_spec(self, spec, title):
        
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        figure = display.specshow(spec, x_axis='time', 
                                y_axis = 'hz', sr=self.sr,
                                hop_length=self.hop_size, ax=ax
                                )
        ax.set(title=title)
        
        return figure

        
def main():
    sample_rate = 44100.
    duration = 10.
    synthGenerator = ExtraUtils(sample_rate, duration)
    synth_waves = synthGenerator.synth_gen()
    
    # Save 20*3 sample waves as audio files
    save_path = '../audio_outputs/synthesized_mixtures/waves/'
    file_dic = {'sine': synth_waves['sine'][:21],
               'square': synth_waves['square'][:21],
               'saw': synth_waves['saw'][:21]}
    saveAudio = SaveAudioFile(sr=sample_rate)
    saveAudio.wav_file(save_path, file_dic)
    
    mixtures = []
    start = 0
    end = 25
    for s in range(40):
        # Lists of waves as a 1D array of signals mixed
        sines = np.sum(np.stack(synth_waves['sine'][start:end]), axis=0)   
        squares = np.sum(np.stack(synth_waves['square'][start:end]), axis=0)
        saws = np.sum(np.stack(synth_waves['saw'][start:end]), axis=0)
        mixtures.append(sines + squares + saws)
        start += 5 
        end += 5
        
    save_path = '../audio_outputs/synthesized_mixtures/mixtures/'
    saveAudio.mix_file(save_path, mixtures)
    
    mix_noiseW = [] # With White Noise
    mix_noiseB = [] # With Brownian Noise
    for mix in mixtures:
        mix_noiseW.append(NoiseGen.white(mix))
        mix_noiseB.append(NoiseGen.brown(mix))
    
    save_path = '../audio_outputs/synthesized_mixtures/mix_white/'
    saveAudio.mix_file(save_path, mix_noiseW)
    
    save_path = '../audio_outputs/synthesized_mixtures/mix_brownian/'
    saveAudio.mix_file(save_path, mix_noiseB)
    
    
    n_fft = 1024
    win_len = 1024
    hop_size = 512
    mels = 128
    
    featExtractor = Features(sample_rate, n_fft=n_fft,
                             win_len=win_len, hop_len=hop_size
                             )
    Visualizer = SpecVis(sample_rate, hop_size)
    main_path = '../figures/spectrograms/'
    for m in range(40):
        clean_spec = featExtractor.mel_spec(
            mixtures[m], mels=mels, log=True
            )
        img = Visualizer.vis_spec(
            clean_spec, title=f'Log-mel Spectrogram (clean) : {m+1}'
            )
        plt.savefig(main_path+f'clean_mixtures/{m+1}.png')
        
        white_spec = featExtractor.mel_spec(
            mix_noiseW[m], mels=mels, log=True
            )
        img = Visualizer.vis_spec(
            white_spec, title=f'Log-mel Spectrogram (white noise : {m+1}'
            )
        plt.savefig(main_path+f'white_mixtures/{m+1}.png')
        
        brown_spec = featExtractor.mel_spec(
            mix_noiseB[m], mels=mels, log=True
            )
        img = Visualizer.vis_spec(
            brown_spec, title=f'Log-mel Spectrogram (brownian noise : {m+1}'
            )
        plt.savefig(main_path+f'brownian_mixtures/{m+1}.png')

    
    
    

if __name__ == '__main__':
    main()

    
    
    
    
    