# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:53:36 2025

@author: kids
"""


import numpy as np
import nidaqmx, pyvisa, time, os, h5py, sys
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
# import set_Weinschel as wein
from Hardware_Control import SIM921_RT as GRT
from Hardware_Control import read_lakeshore_sensors as LS
from Hardware_Control import setSMASwitch
from Hardware_Control import ctrl_preamp_SR560 as SR560
from Data_Calibration.IQ_mixer_calibrate_data_ysl import main_calib_IQ_DCdata_d, main_calib_IQ_ACdata_d
from scipy.optimize import curve_fit
#sys.path.append('D:/Python_scripts/scraps-master/')
import scraps as scr
import scipy.signal as sig
import lmfit as lf
import scipy.special as sp
from Hardware_Control import set_MiniCAtten, GetBoxAtten
from Hardware_Control import setBoxPortSwitch
import scipy.special as sp
import IQ_mixer_process_twotones as Process2tones
import random

IQ_calibfileDC_path = 'D:/noise_data/IQ_calib/20240325_133843_DC_gain20_RFm16/h5_data/IQ_calib_fit.h5'
IQ_calibfileAC_path = 'D:/noise_data/IQ_calib/20240325_144350_AC_gain200_RFm16/h5_data/IQ_calib_fit.h5'

kb_eV = 8.6173423e-5 # [eV/K] Boltzmann cte
h_eV = 4.135667696e-15 # [eV.s] Planck cte
hbar_eV = h_eV/(2 * np.pi)
N0 = 1.71e10 # [eV/um^3] Single spin electron density of states at the Fermi level (S. Siegel thesis, p44)

mydpi = 100 # Resolution for displayed plots


angle = -0.896
angle = angle-0.4
#angle = -1.64
dIQ_vs_df = 3.327e-5
#L = len(f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise'].keys())

N_noise = 10
print(N_noise)
fs = 1e6/10
Npts = 125000000
S_time = Npts/fs
timedata = np.arange(0, S_time, 1/fs)
nfactor = 100
nperseg = int(Npts/nfactor)
LP_cutoff = 30000
Filter_order=6
#i=0

#print(f"The fitted frequency is {resfit_dict['fres']*1e-6} MHz")
print(f"The Calibration angle is {angle} rad")
dir_path = 'D:/noise_data/IQ_data/'

file_path = filedialog.askopenfilename(title='Choose .h5 file with Sdff data', initialdir=dir_path)

Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
filename_sdff = Sdf_dir + 'TwoTones_Sdff'              
tone1_I = np.array([])
tone1_Q = np.array([])
tone2_I = np.array([])
tone2_Q = np.array([])

with h5py.File(file_path, 'r') as hf:


    for i in range(10):
    
        tone1_I = np.append(tone1_I, hf['tone1_I_demodulated%d'%i][100:])
        tone2_I = np.append(tone2_I, hf['tone2_I_demodulated%d'%i][100:])
        tone1_Q = np.append(tone1_Q, hf['tone1_Q_demodulated%d'%i][100:])
        tone2_Q = np.append(tone2_Q, hf['tone2_Q_demodulated%d'%i][100:])
        print("getting data")


theta1 = np.angle(tone1_I + 1j*tone1_Q).mean()
theta2 = np.angle(tone2_I + 1j*tone2_Q).mean()

#print(f"The measured angle of demodulated I and Q is {theta1} rad at {CarrierTones[0]/1e3} kHz and {theta2} rad at {CarrierTones[1]/1e3} kHz, the calib angle is {angle} rad")

No_scale = False

if No_scale:
    data_noise_onres_freq, data_noise_onres_diss = Process2tones.rotate_noise_IQ2FD_nonescale(tone1_I, tone1_Q, angle)
else:
    data_noise_onres_freq, data_noise_onres_diss = Process2tones.rotate_noise_IQ2FD(tone1_I, tone1_Q, angle, dIQ_vs_df, 255.83e6)
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
    data_noise_offres_freq, data_noise_offres_diss = Process2tones.rotate_noise_IQ2FD_nonescale(tone2_I, tone2_Q, angle)
else:
    data_noise_offres_freq, data_noise_offres_diss = Process2tones.rotate_noise_IQ2FD(tone2_I, tone2_Q, angle, dIQ_vs_df, 255.83e6-240e3)
f_data, Sdff_offres_freq = sig.welch(data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
_, Sdff_offres_diss = sig.welch(data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 



axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', label = 'offres: freq')
axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', label = 'offres: diss')
axs[1].tick_params(axis='y', labelsize=14)
axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
axs[1].tick_params(axis='x', labelsize=14)
axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
axs[1].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
axs[1].relim()
axs[1].set_ylim([1e-20, 1e-15])
axs[1].autoscale_view()

filename = Sdf_dir + 'TwoTones_Sdff'
print('Saving the Plot')
fig.savefig(filename + '.pdf', dpi=100)
h5_plot_data_filepath = h5_output_filepath = os.path.join(Sdf_dir, f'plot_data.h5')

with h5py.File(h5_plot_data_filepath, 'w') as hf:
     hf.create_dataset('freq', data = f_data[:int(LP_cutoff*S_time/nfactor)])
     hf.create_dataset('onres_freq_noise',  data=Sdff_onres_freq[:int(LP_cutoff*S_time/nfactor)])
     hf.create_dataset('onres_diss_noise',data =  Sdff_onres_diss[:int(LP_cutoff*S_time/nfactor)])
     hf.create_dataset('offres_freq_noise',data =  Sdff_offres_freq[:int(LP_cutoff*S_time/nfactor)])
     hf.create_dataset('offres_diss_noise',data =  Sdff_offres_diss[:int(LP_cutoff*S_time/nfactor)])



#return I, Q
    