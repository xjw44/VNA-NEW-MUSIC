# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 10:48:04 2025

@author: kids
"""

import numpy as np
import os, h5py, sys
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fftpack import fft, ifft
import scipy.signal as sig
import scipy as sc
import lmfit as lf
import scipy.special as sp
from PyPDF2 import PdfFileMerger
import glob, itertools
import matplotlib as mpl
from scipy.stats import kstest
from shutil import copy
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from scipy.signal import csd
from scipy.signal import coherence
import pandas as pd
import random
import IQ_mixer_process_twotones as process2tones
V_KIDs = 3224 # Inductor volume, in um^3
kb_eV = 8.6173423e-5 # [eV/K] Boltzmann cte
h_eV = 4.135667696e-15 # [eV.s] Planck cte
hbar_eV = h_eV/(2 * np.pi)
N0 = 1.71e10 # [eV/um^3] Single spin electron density of states at the Fermi level (S. Siegel thesis, p44)
eV2J = 1.602176634e-19
mydpi = 100 # Resolution for displayed plots
f_lims = [10, 1e5] # in Hz, frequency limits of the Sdff
zoom = 5 # factor to multiply the noise blob to make it more visible
step = 5000 # factor to plot less points
dir_path = 'D:/noise_data/IQ_data/'
root = tk.Tk()
I_demodulated = True
Q_demodulated = True
file_path = filedialog.askopenfilename(title='Choose .h5 file with Sdff data', initialdir=dir_path)
filename = file_path.split('/')[:-3]
root.destroy()
main_dict = {}
meas_dict={}
Carriers_dict = {'Modulation_Tones':np.zeros([2])}
Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
resfit_dict = {'Qi':np.zeros([1]), 'Qc':np.zeros([1]), 'Qr':np.zeros([1]), 'fres':np.zeros([1])}
resfit_dict_tone2 = {'Qi':np.zeros([1]), 'Qc':np.zeros([1]), 'Qr':np.zeros([1]), 'fres':np.zeros([1])}
#h5_output_filepath = os.path.join(Sdf_dir, f'IQ_demodulated.h5')
Y_limit = [1e-20, 1e-15]
CarrierTones = np.array([120, 370])*1e3
fres= 255.83e6
df_on_off_res = CarrierTones[1]-CarrierTones[0]
angle = -1.346#"f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['angle'][()]#-0.896"
#1angle = -angle+0.2
#angle = -1.64
dIQ_vs_df = 3.327e-5#f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['dIQ_vs_df'][()]#3.327*10^-5

fs = 1.25e6/10#meas_dict['DAQ_freq']/10
Npts = 125000000#$meas_dict['DAQ_Npts']#*10#125000000
S_time = Npts/fs
timedata = np.arange(0, S_time, 1/fs)
nfactor = 10
nperseg = int(Npts/nfactor)
LP_cutoff = 30000
Filter_order=6
#i=0

print(f"The fitted frequency is {resfit_dict['fres']*1e-6} MHz")
print(f"The Calibration angle is {angle} rad")

Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
filename_sdff = Sdf_dir + 'TwoTones_Sdff'        
num_data_runs = 10      
tone1_I = np.array([])
tone1_Q = np.array([])
tone2_I = np.array([])
tone2_Q = np.array([])

cmap = plt.get_cmap('viridis')

colors = cmap(np.linspace(0, 1, num_data_runs))
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)

if I_demodulated and Q_demodulated:
    with h5py.File(file_path, 'r') as hf:
        for i in range(num_data_runs):
            assert os.path.exists(file_path), "h5 file does not exist"
            print(f"Loading pre-existing demodulated data from: {file_path}")
            try:
                with h5py.File(file_path, 'r') as hf:
                    I_1 = np.array(hf['tone1_I_demodulated%d'%i][1000:])
                    I_2 = np.array(hf['tone2_I_demodulated%d'%i][1000:])
                    Q_1 = np.array(hf['tone1_Q_demodulated%d'%i][1000:])
                    Q_2 = np.array(hf['tone2_Q_demodulated%d'%i][1000:])
                    tone1_I = np.append(tone1_I, I_1)
                    tone2_I = np.append(tone2_I, I_2)
                    tone1_Q = np.append(tone1_Q, Q_1)
                    tone2_Q = np.append(tone2_Q, Q_2)
                    
                    I_1_filtered = process2tones.butter_lowpass_filter(I_1, 1, fs, 3)
                    I_2_filtered = process2tones.butter_lowpass_filter(I_2, 1, fs, 3)
                    Q_1_filtered = process2tones.butter_lowpass_filter(Q_1, 1, fs, 3)
                    Q_2_filtered = process2tones.butter_lowpass_filter(Q_2, 1, fs, 3)


                    
                    # Create a colormap
                    
                    # Map array positions to colors
                    axs[0][0].scatter(I_1_filtered[::15000][100:], I_2_filtered[::15000][100:],c=colors[i])
                    axs[0][0].set_xlabel('I_1', fontsize = 8)
                    axs[0][0].set_ylabel('I_2', fontsize = 8)
                    
                    #axs[0][0].colorbar()
                    #axs[1].plot(, ,".",linewidth=0.5, label = 'off vs on', color = 'orange', )
                    axs[0][0].axis('equal')
                    
                    # Map array positions to colors
                    axs[0][1].scatter(Q_1_filtered[::15000][100:], Q_2_filtered[::15000][100:],c=colors[i])
                    axs[0][1].set_xlabel('Q_1', fontsize = 8)
                    axs[0][1].set_ylabel('Q_2', fontsize = 8)
                    #axs[0][1].colorbar()

                    #axs[1].plot(, ,".",linewidth=0.5, label = 'off vs on', color = 'orange', )
                    axs[0][1].axis('equal')

                    
                    theta1 = np.angle(I_1.mean() + 1j*Q_1.mean())
                    theta2 = np.angle(I_2.mean() + Q_2.mean())
                    data_noise_onres_gain, data_noise_onres_phase = process2tones.rotate_noise_IQ2FD_nonescale(I_1,Q_1, theta1)
        
                    data_noise_offres_gain, data_noise_offres_phase = process2tones.rotate_noise_IQ2FD_nonescale(I_2,Q_2, theta2)
                    
                    
                    #We then want to extract noise from them in that direction. 
                    data_noise_onres_gain_filtered = process2tones.butter_lowpass_filter(data_noise_onres_gain, 1, fs, 3)
                    data_noise_onres_phase_filtered = process2tones.butter_lowpass_filter(data_noise_onres_phase, 1, fs, 3)
                    data_noise_offres_gain_filtered = process2tones.butter_lowpass_filter(data_noise_offres_gain, 1, fs, 3)
                    data_noise_offres_phase_filtered = process2tones.butter_lowpass_filter(data_noise_offres_phase, 1, fs, 3)
                    
                    
                    # Create a colormap
                    
                    # Map array positions to colors
                    axs[1][0].scatter(data_noise_onres_gain_filtered[::15000], data_noise_offres_gain_filtered[::15000],c=colors[i], label = "Gain vs Gain")
                    #axs[1].plot(, ,".",linewidth=0.5, label = 'off vs on', color = 'orange', )
                    axs[1][0].axis('equal')
                    axs[1][0].set_xlabel('Gain_On', fontsize = 8)
                    axs[1][0].set_ylabel('Gain_Off', fontsize = 8)
                    #axs[1][0].colorbar()

                    
                    # Map array positions to colors
                    axs[1][1].scatter(data_noise_onres_phase_filtered[::15000], data_noise_offres_phase_filtered[::15000],c=colors[i])
                    #axs[1].plot(, ,".",linewidth=0.5, label = 'off vs on', color = 'orange', )
                    axs[1][1].set_xlabel('Phase_On', fontsize = 8)
                    axs[1][1].set_ylabel('Phase_Off', fontsize = 8)
                    axs[1][1].axis('equal')
                    #axs[1][1].colorbar()


            except Exception as e:
                print(f"Error loading H5 file: {e}.")


   

theta1 = np.angle(tone1_I.mean() + 1j*tone1_Q.mean())
theta2 = np.angle(tone2_I.mean() + 1j*tone2_Q.mean())

#We start by rotating the I and Q into the gain and phase direction











cleaned_data_noise_onres_gain, alpha_est_gain = process2tones.extract_noise(data_noise_onres_gain, data_noise_offres_gain, fs, 1, 3)
cleaned_data_noise_onres_phase, alpha_est_phase = process2tones.extract_noise(data_noise_onres_phase, data_noise_offres_phase, fs, 1, 3)

cleaned_data_onres_I, cleaned_data_onres_Q = process2tones.rotate_noise_IQ2FD_nonescale(cleaned_data_noise_onres_gain, cleaned_data_noise_onres_phase, -theta1)

cleaned_data_noise_offres_gain, alpha_est_gain2 = process2tones.extract_noise(data_noise_offres_gain, data_noise_onres_gain, fs, 1, 3)
cleaned_data_noise_offres_phase, alpha_est_phase2 = process2tones.extract_noise(data_noise_offres_phase, data_noise_onres_phase, fs, 1, 3)

cleaned_data_offres_I, cleaned_data_offres_Q = process2tones.rotate_noise_IQ2FD_nonescale(cleaned_data_noise_offres_gain, cleaned_data_noise_offres_phase, -theta2)



fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
axs[0].plot(timedata[1000:],cleaned_data_noise_onres_gain, linewidth=2, color='C0', label = 'onres: gain')
axs[0].plot(timedata[1000:], cleaned_data_noise_onres_phase, linewidth=2, color='C1', label = 'onres: phase')
axs[0].tick_params(axis='y', labelsize=14)
axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
axs[0].tick_params(axis='x', labelsize=14)
axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
axs[0].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
axs[0].relim()
#axs[0].set_ylim([1e-20, 1e-15])
axs[0].autoscale_view()  




No_scale = False

if No_scale:
    data_noise_onres_freq, data_noise_onres_diss = process2tones.rotate_noise_IQ2FD_nonescale(cleaned_data_onres_I, cleaned_data_onres_Q, angle)
else:
    data_noise_onres_freq, data_noise_onres_diss = process2tones.rotate_noise_IQ2FD(cleaned_data_onres_I, cleaned_data_onres_Q, angle, dIQ_vs_df, fres)
f_data, Sdff_onres_freq = sig.welch(data_noise_onres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
_, Sdff_onres_diss = sig.welch(data_noise_onres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 

   
# data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(tone1_I, tone1_Q, angle, dIQ_vs_df, resfit_dict['fres'])
# f_data, Sdff_onres_freq = sig.welch(data_noise_onres_freq, fs=fs/2, window='hann', nperseg=nperseg, return_onesided = True)
# _, Sdff_onres_diss = sig.welch(data_noise_onres_diss, fs=fs/2, window='hann', nperseg=nperseg, return_onesided = True)

# LP_cutoff=100000
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_onres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', label = 'onres: freq')
axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_onres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', label = 'onres: diss')
axs[0].tick_params(axis='y', labelsize=14)
axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
axs[0].tick_params(axis='x', labelsize=14)
axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
axs[0].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
axs[0].relim()
axs[0].set_ylim([1e-20, 1e-15])
axs[0].autoscale_view()  
#figtitle = '%s:  %dmK  fitres %sMHz'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), meas_dict['freq_center_str'])
#fig.text(0.5, 0.95, figtitle, fontsize=12, horizontalalignment='center', verticalalignment='top')
if No_scale:
    data_noise_offres_freq, data_noise_offres_diss = process2tones.rotate_noise_IQ2FD_nonescale(cleaned_data_onres_I, cleaned_data_onres_Q, angle)
else:
    data_noise_offres_freq, data_noise_offres_diss = process2tones.rotate_noise_IQ2FD(cleaned_data_offres_I, cleaned_data_offres_Q, angle, dIQ_vs_df, fres + df_on_off_res)
f_data, Sdff_offres_freq = sig.welch(data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
_, Sdff_offres_diss = sig.welch(data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 



axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', alpha = 0.2, label = 'offres: freq')
axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', alpha = 0.2, label = 'offres: diss')
axs[1].tick_params(axis='y', labelsize=14)
axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
axs[1].tick_params(axis='x', labelsize=14)
axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
axs[1].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
axs[1].relim()
axs[1].set_ylim([1e-20, 1e-15])
axs[1].autoscale_view()

#%%
No_scale = False

if No_scale:
    data_noise_onres_freq, data_noise_onres_diss = process2tones.rotate_noise_IQ2FD_nonescale(tone1_I, tone1_Q, angle)
else:
    data_noise_onres_freq, data_noise_onres_diss = process2tones.rotate_noise_IQ2FD(tone1_I, tone1_Q, angle, dIQ_vs_df, fres)
f_data, Sdff_onres_freq = sig.welch(data_noise_onres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
_, Sdff_onres_diss = sig.welch(data_noise_onres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 

   
# data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(tone1_I, tone1_Q, angle, dIQ_vs_df, resfit_dict['fres'])
# f_data, Sdff_onres_freq = sig.welch(data_noise_onres_freq, fs=fs/2, window='hann', nperseg=nperseg, return_onesided = True)
# _, Sdff_onres_diss = sig.welch(data_noise_onres_diss, fs=fs/2, window='hann', nperseg=nperseg, return_onesided = True)

# LP_cutoff=100000
# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
# fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_onres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha = 0.2, color='C0', label = 'onres: freq non cleaned')
axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_onres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha = 0.2, color='C1', label = 'onres: diss non cleaned')
axs[0].tick_params(axis='y', labelsize=14)
axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
axs[0].tick_params(axis='x', labelsize=14)
axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
axs[0].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
axs[0].relim()
axs[0].set_ylim([1e-20, 1e-15])
axs[0].autoscale_view()  
# figtitle = '%s:  %dmK  fitres %sMHz'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), meas_dict['freq_center_str'])
# fig.text(0.5, 0.95, figtitle, fontsize=12, horizontalalignment='center', verticalalignment='top')
if No_scale:
    data_noise_offres_freq, data_noise_offres_diss = process2tones.rotate_noise_IQ2FD_nonescale(tone2_I, tone2_Q, angle)
else:
    data_noise_offres_freq, data_noise_offres_diss = process2tones.rotate_noise_IQ2FD(tone2_I, tone2_Q, angle, dIQ_vs_df, fres + df_on_off_res)
f_data, Sdff_offres_freq = sig.welch(data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
_, Sdff_offres_diss = sig.welch(data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 



axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', label = 'offres: freq non cleaned')
axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', label = 'offres: diss non cleaned')
axs[1].tick_params(axis='y', labelsize=14)
axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
axs[1].tick_params(axis='x', labelsize=14)
axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
axs[1].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
axs[1].relim()
axs[1].set_ylim([1e-20, 1e-15])
axs[1].autoscale_view()
