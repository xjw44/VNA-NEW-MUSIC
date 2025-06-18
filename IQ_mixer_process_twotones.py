 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:22:16 2024

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

def load_Sdff(dir_path, device_allres):
    root = tk.Tk()
    file_path = filedialog.askopenfilename(title='Choose .h5 file with Sdff data', initialdir=dir_path)
    root.destroy()
    print("---  Opening datafile %s  ---"%(file_path)) 
    
    # N_noise = 50 # Num of consecutive noise measurements. should be given by the file
    main_dict = {}
    meas_dict = {}
    resfit_dict = {}
    calib_dict = {}
    # f = h5py.File(file_path, 'r')
    with h5py.File(file_path, 'r') as f:
        N_load = len(f.keys())
        key_BBtemp = list(f.keys())[0]
        N_temp = len(f[key_BBtemp].keys())
        key_temp_0 = list(f[key_BBtemp].keys())[0]
        N_pwr = len(f[key_BBtemp][key_temp_0].keys())
        key_pwr_0 = list(f[key_BBtemp][key_temp_0].keys())[0]
        N_data = len(f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['noise']['I_onres0'][()])
        F_data = len(f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['Sdff']['f'][()])
        
        for key in f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['data'].attrs.keys():
            main_dict[key] = f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['data'].attrs[key]
        for key in f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['data']['meas_dict'].attrs.keys():
            main_dict[key] = f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['data']['meas_dict'].attrs[key]    
        for key in f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['noise']['meas_dict'].attrs.keys():
            meas_dict[key] = f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['noise']['meas_dict'].attrs[key]
        # L = len(f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['Sdff'].keys())
        # N_noise = int((L-3)/4-1)
        L = len(f[key_BBtemp][key_temp_0][key_pwr_0]['res_scan']['noise'].keys())
        N_noise = int((L-1)/4)
        
    main_dict['res_ID'] = np.argmin(np.abs(device_allres - main_dict['freq_center']))
    data_dict = {'noise_onres_I':np.zeros([N_temp, N_pwr, N_noise, N_data]), 'noise_onres_Q':np.zeros([N_temp, N_pwr, N_noise, N_data]), 
                 'noise_offres_I':np.zeros([N_temp, N_pwr, N_noise, N_data]), 'noise_offres_Q':np.zeros([N_temp, N_pwr, N_noise, N_data]), 
                 'Sdff_onres_freq':np.zeros([N_temp, N_pwr, N_noise, F_data]), 'Sdff_onres_diss':np.zeros([N_temp, N_pwr, N_noise, F_data]), 
                 'Sdff_offres_freq':np.zeros([N_temp, N_pwr, N_noise, F_data]), 'Sdff_offres_diss':np.zeros([N_temp, N_pwr, N_noise, F_data]), 
                 'noise_I_mean':np.zeros([N_temp, N_pwr, N_noise]), 'noise_I_var':np.zeros([N_temp, N_pwr, N_noise]), 
                 'noise_Q_mean':np.zeros([N_temp, N_pwr, N_noise]), 'noise_Q_var':np.zeros([N_temp, N_pwr, N_noise]),  
                 'noise_select_index':np.zeros([N_temp, N_pwr, N_noise]),  
                 'f':np.zeros([N_temp, N_pwr, F_data]), 'FP_temp':np.zeros([N_temp, N_pwr]), 'pwr@device':np.zeros([N_temp, N_pwr]), 
                 'N_temp':N_temp, 'N_pwr':N_pwr, 'N_noise':N_noise, 'N_data':N_data, 'df_onoff_res':main_dict['df_onoff_res']}
    
    resfit_dict = {'Qi':np.zeros([N_temp, N_pwr]), 'Qc':np.zeros([N_temp, N_pwr]), 'Qr':np.zeros([N_temp, N_pwr]), 
                   'fres':np.zeros([N_temp, N_pwr])}
    calib_dict = {'angle':np.zeros([N_temp, N_pwr]), 'dIQ_vs_df':np.zeros([N_temp, N_pwr])}
    fs = meas_dict['DAQ_freq']
    nperseg = int(meas_dict['DAQ_Npts']/50) # 2024/04/02 FD changed from 10 to 50 to get less noise
    
    with h5py.File(file_path, 'r') as f:
        for i_tmp, key_FPtemp in enumerate(f[key_BBtemp].keys()):
            for i_pwr, key_power in enumerate(f[key_BBtemp][key_FPtemp].keys()):
                for key in resfit_dict.keys():
                    resfit_dict[key][i_tmp][i_pwr] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['fit'].attrs[key]
                calib_dict['angle'][i_tmp][i_pwr] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['calib']['angle'][()]
                calib_dict['dIQ_vs_df'][i_tmp][i_pwr] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['calib']['dIQ_vs_df'][()]
                
                # data_dict['f'][i_tmp][i_pwr] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['Sdff']['f'][()]
                data_dict['FP_temp'][i_tmp][i_pwr] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['noise']['meas_dict'].attrs['FPtemp_start']
                data_dict['pwr@device'][i_tmp][i_pwr] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['noise']['meas_dict'].attrs['pwr@device']
                for i_noise in range(N_noise):
                    data_dict['noise_onres_I'][i_tmp][i_pwr][i_noise] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['noise']['I_onres%d'%i_noise][()]
                    data_dict['noise_onres_Q'][i_tmp][i_pwr][i_noise] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['noise']['Q_onres%d'%i_noise][()]
                    data_dict['noise_I_mean'][i_tmp][i_pwr][i_noise] = np.mean(data_dict['noise_onres_I'][i_tmp][i_pwr][i_noise])
                    data_dict['noise_I_var'][i_tmp][i_pwr][i_noise] = np.var(data_dict['noise_onres_I'][i_tmp][i_pwr][i_noise])
                    data_dict['noise_Q_mean'][i_tmp][i_pwr][i_noise] = np.mean(data_dict['noise_onres_Q'][i_tmp][i_pwr][i_noise])
                    data_dict['noise_Q_var'][i_tmp][i_pwr][i_noise] = np.var(data_dict['noise_onres_Q'][i_tmp][i_pwr][i_noise])
                    
                    data_dict['noise_offres_I'][i_tmp][i_pwr][i_noise] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['noise']['I_offres%d'%i_noise][()]
                    data_dict['noise_offres_Q'][i_tmp][i_pwr][i_noise] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['noise']['Q_offres%d'%i_noise][()]
                    
                    data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(data_dict['noise_onres_I'][i_tmp][i_pwr][i_noise], data_dict['noise_onres_Q'][i_tmp][i_pwr][i_noise], calib_dict['angle'][i_tmp][i_pwr], calib_dict['dIQ_vs_df'][i_tmp][i_pwr], resfit_dict['fres'][i_tmp][i_pwr])
                    data_noise_offres_freq, data_noise_offres_diss = rotate_noise_IQ2FD(data_dict['noise_offres_I'][i_tmp][i_pwr][i_noise], data_dict['noise_offres_Q'][i_tmp][i_pwr][i_noise], calib_dict['angle'][i_tmp][i_pwr], calib_dict['dIQ_vs_df'][i_tmp][i_pwr], resfit_dict['fres'][i_tmp][i_pwr]) #+data_dict['df_onoff_res'])
                    
                    data_dict['f'][i_tmp][i_pwr], data_dict['Sdff_onres_freq'][i_tmp][i_pwr][i_noise] = sig.welch(data_noise_onres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
                    _, data_dict['Sdff_onres_diss'][i_tmp][i_pwr][i_noise] = sig.welch(data_noise_onres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
                    _, data_dict['Sdff_offres_freq'][i_tmp][i_pwr][i_noise] = sig.welch(data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
                    _, data_dict['Sdff_offres_diss'][i_tmp][i_pwr][i_noise] = sig.welch(data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
                # for i_noise in range(N_noise):
                #     data_dict['Sdff_onres_freq'][i_tmp][i_pwr][i_noise] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['Sdff']['PSD_onres_freq%d'%i_noise][()]
                #     data_dict['Sdff_onres_diss'][i_tmp][i_pwr][i_noise] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['Sdff']['PSD_onres_diss%d'%i_noise][()]
                #     data_dict['Sdff_offres_freq'][i_tmp][i_pwr][i_noise] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['Sdff']['PSD_offres_freq%d'%i_noise][()]
                #     data_dict['Sdff_offres_diss'][i_tmp][i_pwr][i_noise] = f[key_BBtemp][key_FPtemp][key_power]['res_scan']['Sdff']['PSD_offres_diss%d'%i_noise][()]

                    
    main_dict['plot_dir'] = file_path[:-36] + 'Plots/'
    meas_dict['str_title_res'] = '%s:  %dmK  %ddBm'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), meas_dict['pwr@device'])
    
    return main_dict, meas_dict, resfit_dict, calib_dict, data_dict

def rotate_noise_IQ2FD(noise_dictI, noise_dictQ, angle, dIQ_vs_df, freq_center):  

    noise = noise_dictI + 1j * noise_dictQ
    noise_centered = noise - np.mean(noise)
    noise_centered_rot = noise_centered * np.exp(-1j*angle)
    data_noise_df_f = noise_centered_rot / dIQ_vs_df / freq_center
    
    return np.imag(data_noise_df_f), np.real(data_noise_df_f)

def rotate_noise_IQ2FD_nonescale(noise_dictI, noise_dictQ, angle):  

    noise = noise_dictI + 1j * noise_dictQ
    noise_centered = noise - np.mean(noise_dictI)-1j*np.mean(noise_dictQ)
    noise_centered_rot = noise_centered * np.exp(1j*angle)
    # data_noise_df_f = noise_centered_rot / dIQ_vs_df / freq_center
    
    return np.real(noise_centered_rot), np.imag(noise_centered_rot)


    
def phase_scale(I, Q):  

    phase = np.arctan2(Q, I)
    phase_mean = np.median(phase)
    if np.abs(phase_mean) > np.abs(np.abs(phase_mean)-np.pi):
        phase = np.fmod(phase + 2*np.pi, 2*np.pi)
    
    return phase
    
    
def plot_noise_time_blob(main_dict, meas_dict, resfit_dict, calib_dict, data_dict, filenames, saveplots=False):
    '''
    Plot the noise data.
    Plot 11: amplitude and phase timestreams
    Plot 12: Noise data blob on the IQ VNA circle
    '''
    N_noise = data_dict['N_noise']
    N_pwr = data_dict['N_pwr']
    N_temp = data_dict['N_temp']
    Non = 6
    Noff = 2
    Ionres_noise_plot =[]
    Qonres_noise_plot =[]
    Ioffres_noise_plot =[]
    Qoffres_noise_plot =[]
    
    for i_pwr in range(N_pwr):
        
        for i_temp in range(N_temp):    
            fig1, ax1 = plt.subplots(Noff+Non+1, sharex=True, gridspec_kw=dict(height_ratios=np.hstack([Non*[1], 1/4, Noff*[1]])*12/(Non+Noff+1/4), hspace=0), figsize=(12, 9))
            fig1.subplots_adjust(bottom=0.09, top=0.9, right=0.97, left=0.12)
            ax1[Non].remove()
            T = data_dict['FP_temp'][i_temp][i_pwr]
            pwr = data_dict['pwr@device'][i_temp][i_pwr]
            fres = resfit_dict['fres'][i_temp][i_pwr]
            
            for i_noise in range(N_noise):
                Ionres_noise_plot = np.append(Ionres_noise_plot, data_dict['noise_onres_I'][i_temp][i_pwr][i_noise])
                Qonres_noise_plot = np.append(Qonres_noise_plot, data_dict['noise_onres_Q'][i_temp][i_pwr][i_noise])
                Ioffres_noise_plot = np.append(Ioffres_noise_plot, data_dict['noise_offres_I'][i_temp][i_pwr][i_noise])
                Qoffres_noise_plot = np.append(Qoffres_noise_plot, data_dict['noise_offres_Q'][i_temp][i_pwr][i_noise])
            
            time_totle=len(Ionres_noise_plot)/meas_dict['DAQ_freq']
            Ionres_noise_plot = Ionres_noise_plot[::step]
            Qonres_noise_plot = Qonres_noise_plot[::step]
            Ioffres_noise_plot = Ioffres_noise_plot[::step]
            Qoffres_noise_plot = Qoffres_noise_plot[::step]
            onres_Amp_plot = np.sqrt(np.square(Ionres_noise_plot)+np.square(Qonres_noise_plot))
            onres_Ang_plot = phase_scale(Ionres_noise_plot, Qonres_noise_plot)
            offres_Amp_plot = np.sqrt(np.square(Ioffres_noise_plot)+np.square(Qoffres_noise_plot))
            offres_Ang_plot = phase_scale(Ioffres_noise_plot, Qoffres_noise_plot)
            onres_freq_plot0, onres_diss_plot0 = rotate_noise_IQ2FD_nonescale(Ionres_noise_plot, Qonres_noise_plot, calib_dict['angle'][i_temp][i_pwr])
            onres_freq_plot, onres_diss_plot = rotate_noise_IQ2FD(Ionres_noise_plot, Qonres_noise_plot, calib_dict['angle'][i_temp][i_pwr], calib_dict['dIQ_vs_df'][i_temp][i_pwr], resfit_dict['fres'][i_temp][i_pwr])
            # offres_freq_plot, offres_diss_plot = rotate_noise_IQ2FD(Ioffres_noise_plot, Qoffres_noise_plot, calib_dict['angle'][i_temp][i_pwr], calib_dict['dIQ_vs_df'][i_temp][i_pwr], resfit_dict['fres'][i_temp][i_pwr]+data_dict['df_onoff_res'])
            t_for_plot = np.linspace(0, time_totle, len(Ionres_noise_plot))
            
            I_outmean_i = KsNormDetect(data_dict['noise_I_mean'][i_temp][i_pwr])
            I_outvar_i = KsNormDetect(data_dict['noise_I_var'][i_temp][i_pwr])
            Q_outmean_i = KsNormDetect(data_dict['noise_Q_mean'][i_temp][i_pwr])
            Q_outvar_i = KsNormDetect(data_dict['noise_Q_var'][i_temp][i_pwr])
        
            noise_original_index = [i for i in range(1, N_noise)]
            # noise_abnormal_index = set(I_outmean_i).union(set(I_outvar_i), set(Q_outmean_i), set(Q_outvar_i), set(f_outmean_i), set(f_outvar_i), set(d_outmean_i), set(d_outvar_i))
            noise_abnormal_index = set(I_outmean_i).union(set(I_outvar_i), set(Q_outmean_i), set(Q_outvar_i))
            noise_select_index = set(noise_original_index).difference(noise_abnormal_index)
            data_dict['noise_select_index'][i_temp][i_pwr][0] = len(noise_select_index)
            i=1
            for key in noise_select_index:
                data_dict['noise_select_index'][i_temp][i_pwr][i] = int(key)
                i+=1
            print('data selection: delete%s'%str(noise_abnormal_index))
            
            label1_ornot=0
            label2_ornot=0
            
            while len(noise_abnormal_index)>0:
                if len(I_outmean_i)>0:
                    if label1_ornot>0:
                        ax1[0].vlines([I_outmean_i*time_totle/N_noise, (I_outmean_i+1)*time_totle/N_noise], np.mean(Ionres_noise_plot)-3*np.var(Ionres_noise_plot), np.mean(Ionres_noise_plot)+3*np.var(Ionres_noise_plot), color='black', rasterized=True)
                    else:
                        ax1[0].vlines([I_outmean_i*time_totle/N_noise, (I_outmean_i+1)*time_totle/N_noise], np.mean(Ionres_noise_plot)-3*np.var(Ionres_noise_plot), np.mean(Ionres_noise_plot)+3*np.var(Ionres_noise_plot), color='black', label = 'mean abnormal chunks', rasterized=True)
                        label1_ornot=1
                if len(I_outvar_i)>0:  
                    if label2_ornot>0:
                        ax1[0].vlines([I_outvar_i*time_totle/N_noise, (I_outvar_i+1)*time_totle/N_noise], np.mean(Ionres_noise_plot)-3*np.var(Ionres_noise_plot), np.mean(Ionres_noise_plot)+3*np.var(Ionres_noise_plot), color='black', linestyles='dashed', rasterized=True)
                    else:
                        ax1[0].vlines([I_outvar_i*time_totle/N_noise, (I_outvar_i+1)*time_totle/N_noise], np.mean(Ionres_noise_plot)-3*np.var(Ionres_noise_plot), np.mean(Ionres_noise_plot)+3*np.var(Ionres_noise_plot), color='black', linestyles='dashed', label = 'var abnormal chunks', rasterized=True)   
                        label2_ornot=1
                if len(Q_outmean_i)>0:   
                    if label1_ornot>0:
                        ax1[1].vlines([Q_outmean_i*time_totle/N_noise, (Q_outmean_i+1)*time_totle/N_noise], np.mean(Qonres_noise_plot)-3*np.var(Qonres_noise_plot), np.mean(Qonres_noise_plot)+3*np.var(Qonres_noise_plot), color='black', rasterized=True)
                    else:
                        ax1[1].vlines([Q_outmean_i*time_totle/N_noise, (Q_outmean_i+1)*time_totle/N_noise], np.mean(Qonres_noise_plot)-3*np.var(Qonres_noise_plot), np.mean(Qonres_noise_plot)+3*np.var(Qonres_noise_plot), color='black', label = 'mean abnormal chunks', rasterized=True)
                        label1_ornot=1
                if len(Q_outvar_i)>0:   
                    if label2_ornot>0:
                        ax1[1].vlines([Q_outvar_i*time_totle/N_noise, (Q_outvar_i+1)*time_totle/N_noise], np.mean(Qonres_noise_plot)-3*np.var(Qonres_noise_plot), np.mean(Qonres_noise_plot)+3*np.var(Qonres_noise_plot), color='black', linestyles='dashed', rasterized=True)
                    else:
                        ax1[1].vlines([Q_outvar_i*time_totle/N_noise, (Q_outvar_i+1)*time_totle/N_noise], np.mean(Qonres_noise_plot)-3*np.var(Qonres_noise_plot), np.mean(Qonres_noise_plot)+3*np.var(Qonres_noise_plot), color='black', linestyles='dashed', label = 'var abnormal chunks', rasterized=True)
                        label2_ornot=1
                    
                select_index = data_dict['noise_select_index'][i_temp][i_pwr][1:len(noise_select_index)+1].astype(int)
                I_outmean_i = KsNormDetect(data_dict['noise_I_mean'][i_temp][i_pwr][select_index])
                I_outvar_i = KsNormDetect(data_dict['noise_I_var'][i_temp][i_pwr][select_index])
                Q_outmean_i = KsNormDetect(data_dict['noise_Q_mean'][i_temp][i_pwr][select_index])
                Q_outvar_i = KsNormDetect(data_dict['noise_Q_var'][i_temp][i_pwr][select_index])
                
                noise_original_index = select_index
                # noise_abnormal_index = set(I_outmean_i).union(set(I_outvar_i), set(Q_outmean_i), set(Q_outvar_i), set(f_outmean_i), set(f_outvar_i), set(d_outmean_i), set(d_outvar_i))
                noise_abnormal_index = set(select_index[I_outmean_i]).union(set(select_index[I_outvar_i]), set(select_index[Q_outmean_i]), set(select_index[Q_outvar_i]))
                noise_select_index = set(noise_original_index).difference(noise_abnormal_index)
                data_dict['noise_select_index'][i_temp][i_pwr][0] = len(noise_select_index)
                i=1
                for key in noise_select_index:
                    data_dict['noise_select_index'][i_temp][i_pwr][i] = int(key)
                    i+=1
                print('data selection: delete%s'%str(noise_abnormal_index))
                
            vlines_index = time_totle/N_noise
            if N_noise>10:
                Noise_d_N = 10
                Noise_d_times = int(N_noise/10)
            else:
                Noise_d_N = N_noise
                Noise_d_times = 1
            for i_noise in range(Noise_d_N):
                ax1[0].vlines([i_noise*Noise_d_times*vlines_index], np.mean(Ionres_noise_plot)-5*np.var(Ionres_noise_plot), np.mean(Ionres_noise_plot)+5*np.var(Ionres_noise_plot), color='C%d'%9, rasterized=True)
                ax1[1].vlines([i_noise*Noise_d_times*vlines_index], np.mean(Qonres_noise_plot)-5*np.var(Qonres_noise_plot), np.mean(Qonres_noise_plot)+5*np.var(Qonres_noise_plot), color='C%d'%9, rasterized=True)
                ax1[2].vlines([i_noise*Noise_d_times*vlines_index], np.mean(onres_Amp_plot)-5*np.var(onres_Amp_plot), np.mean(onres_Amp_plot)+5*np.var(onres_Amp_plot), color='C%d'%9, rasterized=True)
                ax1[3].vlines([i_noise*Noise_d_times*vlines_index], np.mean(onres_Ang_plot)-5*np.var(onres_Ang_plot), np.mean(onres_Ang_plot)+5*np.var(onres_Ang_plot), color='C%d'%9, rasterized=True)
                ax1[4].vlines([i_noise*Noise_d_times*vlines_index], np.mean(onres_freq_plot)-5*np.var(onres_freq_plot), np.mean(onres_freq_plot)+5*np.var(onres_freq_plot), color='C%d'%9, rasterized=True)
                ax1[5].vlines([i_noise*Noise_d_times*vlines_index], np.mean(onres_diss_plot)-5*np.var(onres_diss_plot), np.mean(onres_diss_plot)+5*np.var(onres_diss_plot), color='C%d'%9, rasterized=True)
            ax1[0].plot(t_for_plot, Ionres_noise_plot, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)    
            ax1[0].set_ylabel('I [V]', fontsize = 12)
            ax1[1].plot(t_for_plot, Qonres_noise_plot, '.', color='C%d'%1, markersize=3, alpha=1, rasterized=True)
            ax1[1].set_ylabel('Q [V]', fontsize = 12)
            ax1[2].plot(t_for_plot, onres_Amp_plot, '.', color='C%d'%2, markersize=3, alpha=1, rasterized=True)
            ax1[2].set_ylabel('radius [V]', fontsize = 10)
            ax1[3].plot(t_for_plot, onres_Ang_plot, '.', color='C%d'%3, markersize=3, alpha=1, rasterized=True)
            ax1[3].set_ylabel('Phase [rad]', fontsize = 10)
            ax1[4].plot(t_for_plot, onres_freq_plot, '.', color='C%d'%4, markersize=3, alpha=1, rasterized=True)
            ax1[4].set_ylabel('freq(df/f)', fontsize = 12)
            ax1[5].plot(t_for_plot, onres_diss_plot, '.', color='C%d'%5, markersize=3, alpha=1, rasterized=True)
            ax1[5].set_ylabel('diss(df/f)', fontsize = 12)
 
            ax1[7].plot(t_for_plot, Ioffres_noise_plot, '.', color='C%d'%7, markersize=3, alpha=1, rasterized=True)
            ax1[7].vlines([vlines_index, 2*vlines_index, 3*vlines_index, 4*vlines_index, 5*vlines_index], np.mean(Ioffres_noise_plot)-5*np.var(Ioffres_noise_plot), np.mean(Ioffres_noise_plot)+5*np.var(Ioffres_noise_plot), color='C%d'%9, rasterized=True)
            ax1[7].set_ylabel('I [V]', fontsize = 12)
            ax1[8].plot(t_for_plot, Qoffres_noise_plot, '.', color='C%d'%8, markersize=3, alpha=1, rasterized=True)
            ax1[8].vlines([vlines_index, 2*vlines_index, 3*vlines_index, 4*vlines_index, 5*vlines_index], np.mean(Qoffres_noise_plot)-5*np.var(Qoffres_noise_plot), np.mean(Qoffres_noise_plot)+5*np.var(Qoffres_noise_plot), color='C%d'%9, label = 'Distinction line between every ten chunks of sampled data', rasterized=True)
            ax1[8].set_ylabel('Q [V]', fontsize = 12)
            for i in range(Non):
                ax1[i].tick_params(axis='y', labelsize=14)
            for i in range(Noff):
                ax1[i+7].tick_params(axis='y', labelsize=14)
                
            ax1[-1].set_xlabel("Time [s]", fontsize = 16)
            ax1[-1].tick_params(axis='x', labelsize=14)
            fig1.text(0.02, 0.68, ' (on res)', va='center', rotation='vertical', fontsize=16)
            fig1.text(0.02, 0.20, ' (off res)', va='center', rotation='vertical', fontsize=16)
            figtitle = '%s:  %dmK  %ddBm  fitres:%dMHz'%(main_dict['dev_name'], int(T*1000), pwr, (fres/1e6))
            fig1.text(0.5, 0.98, figtitle, fontsize=14, horizontalalignment='center', verticalalignment='top')
            fig1.legend(loc="upper center", bbox_to_anchor=[0.5, 0.96], bbox_transform = plt.gcf().transFigure, ncol=3, shadow=True, fancybox=True, fontsize=12)
            # ax1[0].text(0.5, 0.97, fres + ' (raw noise)', fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
        
            if saveplots:
                filename = ['noisetime_fitres_%sMHz_pwr_%ddBm_T_%dmK'%(meas_dict['freq_center_str'], pwr, int(T*1000))]
                filenames.append(save_plots(main_dict['plot_dir'] + 'res' + str(main_dict['res_ID']) + '/', filename, [fig1]))
                
            fig2, ax2 = plt.subplots(nrows=2, ncols=3, gridspec_kw=dict(hspace=0.2, wspace=0.2), figsize=(17, 10))
            fig2.subplots_adjust(bottom=0.09, top=0.85, right=0.96, left=0.08)
            ax2[0,0].scatter(Ionres_noise_plot, Qonres_noise_plot, s=2, c=t_for_plot, cmap='winter', zorder=120, rasterized=True)
            ax2[1,0].scatter(onres_Amp_plot, onres_Ang_plot, s=2, c=t_for_plot, cmap='winter', zorder=120, rasterized=True)
            ax2[0,1].scatter(onres_freq_plot0, onres_diss_plot0, s=2, c=t_for_plot, cmap='winter', zorder=120, rasterized=True)
            ax2[1,1].scatter(onres_freq_plot, onres_diss_plot, s=2, c=t_for_plot, cmap='winter', zorder=120, rasterized=True)
            ax2[0,2].scatter(Ioffres_noise_plot, Qoffres_noise_plot, s=2, c=t_for_plot, cmap='winter', zorder=120, rasterized=True)
            ax2[1,2].scatter(offres_Amp_plot, offres_Ang_plot, s=2, c=t_for_plot, cmap='winter', zorder=120, rasterized=True)
            
            ax2[0,0].set_xlabel('onres: I [V]', fontsize = 14)
            ax2[0,0].set_ylabel('onres: Q [V]', fontsize = 14)
            ax2[1,0].set_xlabel('onres: Radius [V]', fontsize = 14)
            ax2[1,0].set_ylabel('onres: Phase [rad]', fontsize = 14)
            ax2[0,1].set_xlabel('onres: Freq [V]', fontsize = 14)
            ax2[0,1].set_ylabel('onres: Diss [V]', fontsize = 14)
            ax2[1,1].set_xlabel('onres: Freq(df/f)', fontsize = 14)
            ax2[1,1].set_ylabel('onres: Diss(df/f)', fontsize = 14) 
            ax2[0,2].set_xlabel('offres: I [V]', fontsize = 14)
            ax2[0,2].set_ylabel('offres: Q [V]', fontsize = 14)
            ax2[1,2].set_xlabel('offres: Radius [V]', fontsize = 14)
            ax2[1,2].set_ylabel('offres: Phase [rad]', fontsize = 14)
            
            # fig2.legend(loc="upper center", bbox_to_anchor=[0.5, 0.97], bbox_transform = plt.gcf().transFigure, ncol=3, shadow=True, fancybox=True, fontsize=14)            
            fig2.text(0.5, 0.98, figtitle, fontsize=14, horizontalalignment='center', verticalalignment='top')
            norm = mpl.colors.Normalize(vmin=0, vmax=time_totle)
            cax = fig2.add_axes([0.13, 0.92, 0.75, 0.03])
            cb1 = mpl.colorbar.ColorbarBase(cax, cmap=mpl.cm.winter, norm=norm,orientation='horizontal')
            cb1.set_label('Measurement time [s]', fontsize=14)
            cax.tick_params(axis='x', labelsize=14)
            for i in range(3):
                for k in range(2):
                    ax2[k,i].autoscale(enable=True, axis='both', tight=None)
                    ax2[k,i].tick_params(axis='x', labelsize=10)
                    ax2[k,i].tick_params(axis='y', labelsize=10)
                    ax2[k,i].grid(visible=True, which='both', color='0.65', linestyle='-')
                
            if saveplots:
                filename = ['noiseblob_fitres_%sMHz_pwr_%ddBm_T_%dmK'%(meas_dict['freq_center_str'], pwr, int(T*1000))]
                filenames.append(save_plots(main_dict['plot_dir'] + 'res' + str(main_dict['res_ID']) + '/', filename, [fig2]))    
                # plt.close('all')
            
    return filenames  
          


def truncate_Sdff(f_Sdff, Sdff, f_lims):
    '''
    Truncate the Sdff and frequency data to f_lims
    '''
    i_lims = [np.argmin(np.abs(f_lims[0] - f_Sdff)), np.argmin(np.abs(f_lims[1] - f_Sdff))]
    f_trunc = f_Sdff[i_lims[0]:i_lims[1]]
    Sdff_trunc = Sdff[i_lims[0]:i_lims[1]]
    return f_trunc, Sdff_trunc
    

def linspace_2_logspace(f_Sdff, Sdff):
    
    N_resample_logscale = int((f_Sdff[-1] - f_Sdff[0]) / (f_Sdff[-1] - f_Sdff[-2]))
    f_log_interp = np.logspace(np.log10(f_Sdff[0]), np.log10(f_Sdff[-1]), N_resample_logscale)
    func_interp_Sdff = sc.interpolate.interp1d(f_Sdff, Sdff, kind='quadratic')
    return f_log_interp, func_interp_Sdff(f_log_interp)
    
    
    
def decimate_Sdff(f_Sdff, Sdff, dec_fact=100):
    
    Sdff_dec = sig.decimate(Sdff, dec_fact, ftype='fir')
    f_Sdff_dec = sig.decimate(f_Sdff, dec_fact, ftype='fir')
    return f_Sdff_dec, Sdff_dec


def avg_Sdffs(Sdff_array):
    err_thres = 0.2
    Sdff_median = np.median(Sdff_array,0)
    err = (np.abs(Sdff_array - Sdff_median[None,:]) / Sdff_median[None,:])**2
    selected_i = np.where(np.mean(err,1) < err_thres)[0]
    Sdff_array_clean = Sdff_array[selected_i]
    Sdff_mean_clean = np.mean(Sdff_array_clean,0)
    N_seg = len(Sdff_array_clean)
    print(' A total of %d/%d Sdff spectrums were averaged'%(N_seg, len(Sdff_array)))
    return selected_i, Sdff_mean_clean
    
    
def reso_LPfilter(resfit_dict):
    fres_cutoff = resfit_dict['fres'] / (2 * resfit_dict['Qr'])
    resfit_dict['fres_cutoff'] = fres_cutoff
    return resfit_dict

def fit_tau_Nqp(f_Sdff, SNqp, fres_cutoff, f_range=(1e1, 1e5)):
    def minimize_func(params, f, SNqp):
        # omega = 2*np.pi*f
        reso_filt = np.abs(1 / (1 + 1j * f / params['fres_cutoff']))**2
        qp_filt = (4 * params['tau'] * params['Nqp']) / (1 + (params['tau'] * 2*np.pi*f)**2)
        return qp_filt * reso_filt - SNqp #+ params['Cte']
    
    i_min = np.argmin(np.abs(f_range[0] - f_Sdff))
    i_max = np.argmin(np.abs(f_range[1] - f_Sdff))
    f_Sdff_trunc = f_Sdff[i_min:i_max]
    # tau_reso = Q/(np.pi * fres)
    SNqp_trunc = SNqp[i_min:i_max]
    fit_params = lf.Parameters()
    # fit_params.add('Cte', value=1e-3, min=0, max=100)
    fit_params.add('fres_cutoff', expr='%d'%fres_cutoff)
    fit_params.add('tau', value=100e-6, min=1e-6, max=1e-3)
    fit_params.add('Nqp', value=1e5, min=1e2, max=1e8)
    mini = lf.Minimizer(minimize_func, fit_params, nan_policy='omit', fcn_args=(f_Sdff_trunc, SNqp_trunc))
    out1 = mini.minimize(method='ampgo')
    # out1 = mini.minimize(method='basinhopping')
    # out2 = mini.minimize(method='Nelder', params=out1.params)
    # fit_dict.append(out1.params)
    return out1.params
    

def calc_dnqp_vs_dx(fres, T, MB_filename):
    delta, alpha, f0 = get_MB_params(fres, T, MB_filename)
    omega = 2 * np.pi * fres
    k2 = get_k2(omega, T, delta, N0)
    return 2 / (alpha * k2) 
    

def calc_NEP(f, fres_cutoff, Sdff, delta, V, tau_qp, dnqp_vs_x, eta_pb):
    omega = 2 * np.pi * f
    NEP = np.sqrt(Sdff) * delta * V / eta_pb / tau_qp * dnqp_vs_x
    NEP = NEP * np.sqrt(1 + omega**2 * tau_qp**2) * np.sqrt(1 + f**2 / fres_cutoff**2)
    return NEP
    

def remove_spikes(raw_data):
    '''
    Removes spikes from Sdff signals.  
    The function calculates the median absolute deviation of the signal and remove all spikes
    above 10 times this deviation above the baseline.
    An iterative technique is used to shorten the peaks by half their height at each iteration.
    It is done because when deleting the peak entirely at once, it sometimes also deletes parts of the baseline.
    By doing it this way, it reduces the risks of truncating data other than the peak.
    After the peak is detected, it is deleted by setting it to the mean of the point before and after the peak.
    In spikes matrix, spikes[2] are the left limits of the spikes and spikes[3] are the right limits of the spikes.
    '''
    thres = 0.3 #(sc.stats.median_abs_deviation(raw_data)) * 10
    wlen = 50 # window length (set to 50 pts, might need to be changed for different data sets)
    peaks, dict_peaks = sig.find_peaks(raw_data, prominence=thres, wlen=wlen)
    spikes = np.round(sig.peak_widths(raw_data, peaks, rel_height=0.5)).astype(int)
    print('A total of %d peaks were found and will be corrected'%len(peaks))
    clean_data = raw_data.copy()
    avg_range = 10
    while spikes.any():
        for i in range(len(spikes[2])):
            peak_new_val = (np.mean(clean_data[spikes[2,i]-avg_range : spikes[2,i]]) + 
                            np.mean(clean_data[spikes[3,i]+1 : spikes[3,i]+1+avg_range] )) / 2
            clean_data[spikes[2,i]-1:spikes[3,i]+2] = peak_new_val
        peaks, dict_peaks = sc.signal.find_peaks(clean_data, prominence=thres, wlen=wlen)
        spikes = np.round(sc.signal.peak_widths(clean_data, peaks, rel_height=0.5)).astype(int)
    return clean_data

def plot_all_noise_psd(main_dict, meas_dict, resfit_dict, data_dict, saveplots=False):
    '''
    Plot the noise data.
    Plot 11: amplitude and phase timestreams
    Plot 12: Noise data blob on the IQ VNA circle
    '''
    N_noise = data_dict['N_noise']
    N_pwr = data_dict['N_pwr']
    N_temp = data_dict['N_temp']
    fs = meas_dict['DAQ_freq']
    nperseg = int(meas_dict['DAQ_Npts']/50) # 2024/04/02 FD changed from 10 to 50 to get less noise
    
    
    for i_pwr in range(N_pwr):
        for i_temp in range(N_temp):   
            T = data_dict['FP_temp'][i_temp][i_pwr]
            pwr = data_dict['pwr@device'][i_temp][i_pwr]
            noise_select_len = int(data_dict['noise_select_index'][i_temp][i_pwr][0])
            
            for i_noise in range(N_noise):
                onres_Amp_plot = np.sqrt(np.square(data_dict['noise_onres_I'][i_temp][i_pwr][i_noise])+np.square(data_dict['noise_onres_Q'][i_temp][i_pwr][i_noise]))
                onres_Ang_plot = np.unwrap(np.arctan2(data_dict['noise_onres_Q'][i_temp][i_pwr][i_noise], data_dict['noise_onres_I'][i_temp][i_pwr][i_noise]))
                offres_Amp_plot = np.sqrt(np.square(data_dict['noise_offres_I'][i_temp][i_pwr][i_noise])+np.square(data_dict['noise_offres_Q'][i_temp][i_pwr][i_noise]))
                offres_Ang_plot = np.unwrap(np.arctan2(data_dict['noise_offres_Q'][i_temp][i_pwr][i_noise], data_dict['noise_offres_I'][i_temp][i_pwr][i_noise]))
                
                f_Sdff, onres_Amp_psd = sig.welch(onres_Amp_plot, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
                _, onres_Ang_psd = sig.welch(onres_Ang_plot, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
                _, offres_Amp_psd = sig.welch(offres_Amp_plot, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
                _, offres_Ang_psd = sig.welch(offres_Ang_plot, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
                
                fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
                fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
                axs[0].loglog(f_Sdff, np.abs(onres_Amp_psd), linewidth=2, color='C0', label = 'onres: radius')
                axs[0].loglog(f_Sdff, np.abs(onres_Ang_psd), linewidth=2, color='C1', label = 'onres: phase')
                axs[0].loglog([resfit_dict['fres_cutoff'][i_temp][i_pwr], resfit_dict['fres_cutoff'][i_temp][i_pwr]], [1e-8, 1e-2], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
                axs[1].loglog(f_Sdff, np.abs(offres_Amp_psd), linewidth=2, color='C0', label = 'offres: radius')
                axs[1].loglog(f_Sdff, np.abs(offres_Ang_psd), linewidth=2, color='C1', label = 'offres: phase')
                axs[1].loglog([resfit_dict['fres_cutoff'][i_temp][i_pwr], resfit_dict['fres_cutoff'][i_temp][i_pwr]], [1e-11, 1e-7], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
                
                axs[0].tick_params(axis='y', labelsize=14)
                axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
                axs[0].tick_params(axis='x', labelsize=14)
                axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
                axs[0].set_ylabel(r'PSD  [$Hz^{-1}$]', fontsize = 16)
                axs[1].tick_params(axis='y', labelsize=14)
                axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
                axs[1].tick_params(axis='x', labelsize=14)
                axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
                axs[1].set_ylabel(r'PSD  [$Hz^{-1}$]', fontsize = 16)
                axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
                axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
                axs[0].relim()
                # axs[0].set_ylim(ylims)
                axs[0].autoscale_view()  
                axs[1].relim()
                # axs[1].set_ylim(ylims)
                axs[1].autoscale_view()  
                
                if i_noise not in data_dict['noise_select_index'][i_temp][i_pwr][1:noise_select_len+1]:
                    fig.text(0.5, 0.98, 'Discarded! (by IQ/FD time stream data mean/var 3sigma abnormal chunks)', fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
                else:
                    fig.text(0.5, 0.98, 'Selected!', fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
                    
                if saveplots:
                    file_dir = main_dict['plot_dir'] + 'psd/'  + 'res' + str(main_dict['res_ID']) + '/'
                    os.makedirs(file_dir, exist_ok=True)
                    filename = 'noisepsd_fitres_%sMHz_pwr_%ddBm_T_%dmK_%d'%(meas_dict['freq_center_str'], pwr, int(T*1000), i_noise)
                    fig.savefig(file_dir + filename  + '.png', dpi=mydpi, bbox_inches = 'tight')
                    plt.close('all')



def plot_avg_Sdff_data(data_dict, resfit_dict, main_dict, meas_dict, filenames, dec_fact=100, ylims=[1e-22, 1e-16], saveplots=False):
    N_pwr = data_dict['N_pwr']
    N_temp = data_dict['N_temp']
    N_noise = data_dict['N_noise']
    data_dict['Sdff_onres_freq_mean'] = []
    data_dict['Sdff_onres_diss_mean'] = []
    data_dict['Sdff_offres_freq_mean'] = []
    data_dict['Sdff_offres_diss_mean'] = []
    data_dict['Sdff_onres_freq_clean'] = []
    data_dict['Sdff_onres_diss_clean'] = []
    data_dict['Sdff_offres_freq_clean'] = []
    data_dict['Sdff_offres_diss_clean'] = []

    for i_pwr in range(N_pwr):
        Sdff_onres_freq_mean, Sdff_onres_diss_mean = [], []
        Sdff_offres_freq_mean, Sdff_offres_diss_mean = [], []
        
        for i_temp in range(N_temp):
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(14, 9))
            plt.subplots_adjust(bottom=0.09, top=0.94, right=0.6, left=0.08)
            f_Sdff = data_dict['f'][i_temp][i_pwr]
            T = data_dict['FP_temp'][i_temp][i_pwr]
            pwr = data_dict['pwr@device'][i_temp][i_pwr]
            noise_select_len = int(data_dict['noise_select_index'][i_temp][i_pwr][0])
            
            Sdff_onres_freq_array, Sdff_onres_diss_array = [], []
            Sdff_offres_freq_array, Sdff_offres_diss_array = [], []
            for i in range(N_noise):
                Sdff_onres_freq = data_dict['Sdff_onres_freq'][i_temp][i_pwr][i]
                # f_Sdff_log, Sdff_onres_freq_log = linspace_2_logspace(f_Sdff, Sdff_onres_freq)
                # f_Sdff_dec, Sdff_onres_freq_dec = decimate_Sdff(f_Sdff_log, Sdff_onres_freq_log, dec_fact=dec_fact)
                # f_Sdff_trunc, Sdff_onres_freq_trunc = truncate_Sdff(f_Sdff_dec, Sdff_onres_freq_dec, f_lims)
                Sdff_onres_freq_array.append(Sdff_onres_freq)
                
                Sdff_onres_diss = data_dict['Sdff_onres_diss'][i_temp][i_pwr][i]
                # f_Sdff_log, Sdff_onres_diss_log = linspace_2_logspace(f_Sdff, Sdff_onres_diss)
                # f_Sdff_dec, Sdff_onres_diss_dec = decimate_Sdff(f_Sdff_log, Sdff_onres_diss_log, dec_fact=dec_fact)
                # f_Sdff_trunc, Sdff_onres_diss_trunc = truncate_Sdff(f_Sdff_dec, Sdff_onres_diss_dec, f_lims)
                Sdff_onres_diss_array.append(Sdff_onres_diss)
                
                Sdff_offres_freq = data_dict['Sdff_offres_freq'][i_temp][i_pwr][i]
                Sdff_offres_freq_array.append(Sdff_offres_freq)
                Sdff_offres_diss = data_dict['Sdff_offres_diss'][i_temp][i_pwr][i]
                Sdff_offres_diss_array.append(Sdff_offres_diss)
                
            Sdff_onres_freq_array = np.array(Sdff_onres_freq_array)
            Sdff_onres_diss_array = np.array(Sdff_onres_diss_array)
            Sdff_onres_freq_avg = (np.mean(Sdff_onres_freq_array, 0))
            # Sdff_onres_freq_avg = avg_Sdffs(Sdff_onres_freq_array)
            # Sdff_onres_freq_avg_nopeaks = remove_spikes(np.log10(np.abs(Sdff_onres_freq_avg)))
            # Sdff_onres_freq_avg_nopeaks = 10**(Sdff_onres_freq_avg_nopeaks)
            Sdff_onres_diss_avg = (np.mean(Sdff_onres_diss_array, 0))
            # Sdff_onres_diss_avg = avg_Sdffs(Sdff_onres_diss_array)
            # Sdff_onres_diss_avg_nopeaks = remove_spikes(np.log10(np.abs(Sdff_onres_diss_avg)))
            # Sdff_onres_diss_avg_nopeaks = 10**(Sdff_onres_diss_avg_nopeaks)
            
            Sdff_offres_freq_array = np.array(Sdff_offres_freq_array)
            Sdff_offres_diss_array = np.array(Sdff_offres_diss_array)
            Sdff_offres_freq_avg = (np.mean(Sdff_offres_freq_array, 0))
            Sdff_offres_diss_avg = (np.mean(Sdff_offres_diss_array, 0))
            # Sdff_offres_freq_avg = avg_Sdffs(Sdff_offres_freq_array)
            # Sdff_offres_diss_avg = avg_Sdffs(Sdff_offres_diss_array)
            
            
            axs[0].loglog(f_Sdff, Sdff_onres_freq_avg, 'oC%d'%0, markersize=2, label = '(all) freq data onres')
            axs[0].loglog(f_Sdff, Sdff_onres_diss_avg, 'oC%d'%1, markersize=2, label = '(all) diss data onres')
            axs[1].loglog(f_Sdff, Sdff_offres_freq_avg, 'oC%d'%2, markersize=2, label = '(all) freq data offres')
            axs[1].loglog(f_Sdff, Sdff_offres_diss_avg, 'oC%d'%3, markersize=2, label = '(all) diss data offres')

            Sdff_onres_freq_array, Sdff_onres_diss_array = [], []
            Sdff_offres_freq_array, Sdff_offres_diss_array = [], []
            
            for i in range(noise_select_len):
                select_index = int(data_dict['noise_select_index'][i_temp][i_pwr][1+i])
                Sdff_onres_freq = data_dict['Sdff_onres_freq'][i_temp][i_pwr][select_index]
                # f_Sdff_log, Sdff_onres_freq_log = linspace_2_logspace(f_Sdff, Sdff_onres_freq)
                # f_Sdff_dec, Sdff_onres_freq_dec = decimate_Sdff(f_Sdff_log, Sdff_onres_freq_log, dec_fact=dec_fact)
                # f_Sdff_trunc, Sdff_onres_freq_trunc = truncate_Sdff(f_Sdff_dec, Sdff_onres_freq_dec, f_lims)
                Sdff_onres_freq_array.append(Sdff_onres_freq)
                
                Sdff_onres_diss = data_dict['Sdff_onres_diss'][i_temp][i_pwr][select_index]
                # f_Sdff_log, Sdff_onres_diss_log = linspace_2_logspace(f_Sdff, Sdff_onres_diss)
                # f_Sdff_dec, Sdff_onres_diss_dec = decimate_Sdff(f_Sdff_log, Sdff_onres_diss_log, dec_fact=dec_fact)
                # f_Sdff_trunc, Sdff_onres_diss_trunc = truncate_Sdff(f_Sdff_dec, Sdff_onres_diss_dec, f_lims)
                Sdff_onres_diss_array.append(Sdff_onres_diss)
                
                Sdff_offres_freq = data_dict['Sdff_offres_freq'][i_temp][i_pwr][select_index]
                Sdff_offres_freq_array.append(Sdff_offres_freq)
                Sdff_offres_diss = data_dict['Sdff_offres_diss'][i_temp][i_pwr][select_index]
                Sdff_offres_diss_array.append(Sdff_offres_diss)
                
            Sdff_onres_freq_array = np.array(Sdff_onres_freq_array)
            Sdff_onres_diss_array = np.array(Sdff_onres_diss_array)
            Sdff_onres_freq_avg_selected = (np.mean(Sdff_onres_freq_array, 0))
            # selected_i1, Sdff_onres_freq_avg_selected = avg_Sdffs(Sdff_onres_freq_array)
            # Sdff_onres_freq_avg_nopeaks = remove_spikes(np.log10(np.abs(Sdff_onres_freq_avg)))
            # Sdff_onres_freq_avg_nopeaks = 10**(Sdff_onres_freq_avg_nopeaks)
            Sdff_onres_diss_avg_selected = (np.mean(Sdff_onres_diss_array, 0))
            # selected_i2, Sdff_onres_diss_avg_selected = avg_Sdffs(Sdff_onres_diss_array)
            # Sdff_onres_diss_avg_nopeaks = remove_spikes(np.log10(np.abs(Sdff_onres_diss_avg)))
            # Sdff_onres_diss_avg_nopeaks = 10**(Sdff_onres_diss_avg_nopeaks)
            Sdff_onres_freq_mean.append(Sdff_onres_freq_avg_selected)
            Sdff_onres_diss_mean.append(Sdff_onres_diss_avg_selected)
            
            
            Sdff_offres_freq_array = np.array(Sdff_offres_freq_array)
            Sdff_offres_diss_array = np.array(Sdff_offres_diss_array)
            
            Sdff_offres_freq_avg_selected = np.mean(Sdff_offres_freq_array, 0)
            Sdff_offres_diss_avg_selected = np.mean(Sdff_offres_diss_array, 0)
            Sdff_offres_freq_mean.append(Sdff_offres_freq_avg_selected)
            Sdff_offres_diss_mean.append(Sdff_offres_diss_avg_selected)
            
            axs[0].loglog(f_Sdff, Sdff_onres_freq_avg_selected, '-', color='C%d'%(4), markersize=2, label = '(selected) freq data onres')
            axs[0].loglog(f_Sdff, Sdff_onres_diss_avg_selected, '-', color='C%d'%(5), markersize=2, label = '(selected) diss data onres')
            axs[0].loglog([resfit_dict['fres_cutoff'][i_temp][i_pwr], resfit_dict['fres_cutoff'][i_temp][i_pwr]], [1e-21, 1e-17], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
            axs[1].loglog(f_Sdff, Sdff_offres_freq_avg_selected, '-', color='C%d'%(6), markersize=2, label = '(selected) freq data offres')
            axs[1].loglog(f_Sdff, Sdff_offres_diss_avg_selected, '-', color='C%d'%(7), markersize=2, label = '(selected) diss data offres')
            axs[1].loglog([resfit_dict['fres_cutoff'][i_temp][i_pwr], resfit_dict['fres_cutoff'][i_temp][i_pwr]], [1e-23, 1e-17], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
            
            fig.text(0.5, 0.98, meas_dict['str_title_res'], fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
            axs[0].tick_params(axis='y', labelsize=14)
            axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
            axs[0].tick_params(axis='x', labelsize=14)
            axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
            axs[0].set_ylabel(r'(on res) Sdf/f  [$Hz^{-1}$]', fontsize = 16)
            axs[0].set_title('Noise Sdff_avg:  fitres %sMHz  pwr %ddBm  T %dmK'%(meas_dict['freq_center_str'], pwr, int(T*1000)), fontsize=10)
            axs[1].tick_params(axis='y', labelsize=14)
            axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
            axs[1].tick_params(axis='x', labelsize=14)
            axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
            axs[1].set_ylabel(r'(off res) Sdf/f  [$Hz^{-1}$]', fontsize = 16)
            axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
            axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
            axs[0].relim()
            #axs[0].set_ylim(ylims)
            axs[0].autoscale_view()  
            axs[1].relim()
            #axs[1].set_ylim([1e-23, 1e-17])
            axs[1].autoscale_view() 
        
            if saveplots:
                filename = ['noiseSdffavg_fitres_%sMHz_pwr_%ddBm_T_%dmK'%(meas_dict['freq_center_str'], pwr, int(T*1000))]
                filenames.append(save_plots(main_dict['plot_dir'] + 'res' + str(main_dict['res_ID']) + '/', filename, [fig]))
                plt.close('all')
            
            for i_noise in range(N_noise):
                fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
                fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
                axs[0].loglog(f_Sdff, np.abs(data_dict['Sdff_onres_freq'][i_temp][i_pwr][i_noise]), linewidth=2, color='C0', label = 'onres: freq')
                axs[0].loglog(f_Sdff, np.abs(data_dict['Sdff_onres_diss'][i_temp][i_pwr][i_noise]), linewidth=2, color='C1', label = 'onres: diss')
                axs[0].loglog(f_Sdff, Sdff_onres_freq_avg, 'oC%d'%2, markersize=2, label = '(all) freq data onres')
                axs[0].loglog(f_Sdff, Sdff_onres_diss_avg, 'oC%d'%3, markersize=2, label = '(all) diss data onres')
                axs[0].loglog(f_Sdff, Sdff_onres_freq_avg_selected, '-', color='C4', markersize=2, label = '(selected) freq data onres')
                axs[0].loglog(f_Sdff, Sdff_onres_diss_avg_selected, '-', color='C5', markersize=2, label = '(selected) diss data onres')
                axs[0].loglog([resfit_dict['fres_cutoff'][i_temp][i_pwr], resfit_dict['fres_cutoff'][i_temp][i_pwr]], [1e-21, 1e-17], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
                axs[1].loglog(f_Sdff, np.abs(data_dict['Sdff_offres_freq'][i_temp][i_pwr][i_noise]), linewidth=2, color='C0', label = 'offres: freq')
                axs[1].loglog(f_Sdff, np.abs(data_dict['Sdff_offres_diss'][i_temp][i_pwr][i_noise]), linewidth=2, color='C1', label = 'offres: diss')
                axs[1].loglog(f_Sdff, Sdff_offres_freq_avg, 'oC%d'%2, markersize=2, label = '(all) freq data offres')
                axs[1].loglog(f_Sdff, Sdff_offres_diss_avg, 'oC%d'%3, markersize=2, label = '(all) diss data offres')
                axs[1].loglog(f_Sdff, Sdff_offres_freq_avg_selected, '-', color='C%d'%(4), markersize=2, label = '(selected) freq data offres')
                axs[1].loglog(f_Sdff, Sdff_offres_diss_avg_selected, '-', color='C%d'%(5), markersize=2, label = '(selected) diss data offres')
                axs[1].loglog([resfit_dict['fres_cutoff'][i_temp][i_pwr], resfit_dict['fres_cutoff'][i_temp][i_pwr]], [1e-23, 1e-17], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
                
                axs[0].tick_params(axis='y', labelsize=14)
                axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
                axs[0].tick_params(axis='x', labelsize=14)
                axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
                axs[0].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
                axs[1].tick_params(axis='y', labelsize=14)
                axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
                axs[1].tick_params(axis='x', labelsize=14)
                axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
                axs[1].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
                axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
                axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
                axs[0].relim()
                axs[0].set_ylim(ylims)
                axs[0].autoscale_view()  
                axs[1].relim()
                axs[1].set_ylim([1e-23, 1e-17])
                axs[1].autoscale_view()  
                
                if i_noise not in data_dict['noise_select_index'][i_temp][i_pwr][1:noise_select_len+1]:
                    fig.text(0.5, 0.98, 'Discarded! (by IQ/FD time stream data mean/var 3sigma abnormal chunks)', fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
                else:
                    fig.text(0.5, 0.98, 'Selected!', fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
                
                if saveplots:
                    file_dir = main_dict['plot_dir'] + 'Sdff/all/' + 'res' + str(main_dict['res_ID']) + '/'
                    os.makedirs(file_dir, exist_ok=True)
                    filename = 'noiseSdff_fitres_%sMHz_pwr_%ddBm_T_%dmK_%d'%(meas_dict['freq_center_str'], pwr, int(T*1000), i_noise)
                    fig.savefig(file_dir + filename  + '.png', dpi=mydpi, bbox_inches = 'tight')
                    plt.close('all')
            
        data_dict['Sdff_onres_freq_mean'].append(Sdff_onres_freq_mean)
        data_dict['Sdff_onres_diss_mean'].append(Sdff_onres_diss_mean)
        data_dict['Sdff_offres_freq_mean'].append(Sdff_offres_freq_mean)
        data_dict['Sdff_offres_diss_mean'].append(Sdff_offres_diss_mean)

    data_dict['Sdff_onres_freq_clean'] = np.array(data_dict['Sdff_onres_freq_mean'])
    data_dict['Sdff_onres_diss_clean'] = np.array(data_dict['Sdff_onres_diss_mean'])
    data_dict['Sdff_offres_freq_clean'] = np.array(data_dict['Sdff_offres_freq_mean'])
    data_dict['Sdff_offres_diss_clean'] = np.array(data_dict['Sdff_offres_diss_mean'])
    
    data_dict['Sdff_f_clean'] = f_Sdff
    
    return filenames  


    
def calc_SNqp(data_dict, resfit_dict, MB_filename):
    N_pwr = data_dict['N_pwr']
    N_temp = data_dict['N_temp']
    data_dict['SNqp_onres'] = data_dict['Sdff_onres_freq_clean']*0

    for i_temp in range(N_temp):
        for i_pwr in range(N_pwr):
            fres = resfit_dict['fres'][i_temp][i_pwr]
            T = data_dict['FP_temp'][i_temp][i_pwr]
            delta, alpha, f0 = get_MB_params(fres, T, MB_filename)
            Sdff_freq_onres = data_dict['Sdff_onres_freq_clean'][i_temp][i_pwr] - data_dict['Sdff_onres_diss_clean'][i_temp][i_pwr]
            # Sdff_freq_m_diss_onres = data_dict['Sdff_onres_freq_m_diss_clean'][i_temp]
            # Sdff_freq_m_diss_onres[Sdff_freq_m_diss_onres < 0] = 0
            # data_dict['SNqp_onres'][i_temp] = func_SNqp(fres, T, Sdff_freq_m_diss_onres, delta, V_KIDs)
            Sdff_freq_onres[Sdff_freq_onres < 0] = 0
            data_dict['SNqp_onres'][i_temp][i_pwr] = func_SNqp(fres, T, Sdff_freq_onres, delta, alpha, V_KIDs)
            data_dict['nqp_th'][i_temp][i_pwr] = func_nqp_nogap(delta, T)
    return data_dict


def get_tauqp_Nqp(data_dict, resfit_dict, f_range=(2e1, 5e4)):
    N_pwr = data_dict['N_pwr']
    N_temp = data_dict['N_temp']
    fit_dict = {'tau_qp':np.zeros([N_temp, N_pwr]), 'Nqp':np.zeros([N_temp, N_pwr])}#, 'Cte':np.zeros([N_temp, N_pwr])}
    
    for i_temp in range(N_temp):
        for i_pwr in range(N_pwr):
            SNqp = data_dict['SNqp_onres'][i_temp][i_pwr]
            fres_cutoff = resfit_dict['fres_cutoff'][i_temp][i_pwr]
            results = fit_tau_Nqp(data_dict['f'][i_temp][i_pwr], SNqp, fres_cutoff, f_range=f_range)
            fit_dict['tau_qp'][i_temp][i_pwr] = results['tau'].value
            fit_dict['Nqp'][i_temp][i_pwr] = results['Nqp'].value
            # fit_dict['Cte'][i_temp][i_pwr] = results['Cte'].value
    return fit_dict

def fit_SNqp(data_dict, resfit_dict, fit_dict):
    N_pwr = data_dict['N_pwr']
    N_temp = data_dict['N_temp']
    data_dict['SNqp_fit'] = data_dict['Sdff_onres_freq_clean']*0
    
    for i_temp in range(N_temp):
        for i_pwr in range(N_pwr):
            tau_qp = fit_dict['tau_qp'][i_temp][i_pwr]
            Nqp = fit_dict['Nqp'][i_temp][i_pwr]
            fres_cutoff = resfit_dict['fres_cutoff'][i_temp][i_pwr]
            data_dict['SNqp_fit'][i_temp][i_pwr] = SNqp_fit_func(data_dict['f'][i_temp][i_pwr], tau_qp, Nqp, fres_cutoff)#, Cte)
            
    return data_dict
            

def get_NEP(data_dict, fit_dict, resfit_dict, V_KIDs, MB_filename):
    N_pwr = data_dict['N_pwr']
    N_temp = data_dict['N_temp']
    data_dict['eta_pb'] = 0.5 # varies between 0.9 and 0.5
    data_dict['NEP'] = data_dict['f']*0
    for i_temp in range(N_temp):
        for i_pwr in range(N_pwr):
            fres = resfit_dict['fres'][i_temp][i_pwr]
            fres_cutoff = resfit_dict['fres_cutoff'][i_temp][i_pwr]
            T = data_dict['FP_temp'][i_temp][i_pwr]
            tau_qp = fit_dict['tau_qp'][i_temp][i_pwr]
            delta, alpha, f0 = get_MB_params(fres, T, MB_filename)
            Sdff_freq = data_dict['Sdff_onres_freq_clean'][i_temp][i_pwr]# - data_dict['Sdff_onres_diss_clean'][i_temp][i_pwr]
            dnqp_vs_x = calc_dnqp_vs_dx(fres, T, MB_filename)
            data_dict['NEP'][i_temp][i_pwr] = calc_NEP(data_dict['f'][i_temp][i_pwr], fres_cutoff, np.abs(Sdff_freq), delta*eV2J, V_KIDs, tau_qp, dnqp_vs_x, data_dict['eta_pb'])

    return data_dict


def plot_SNqp_NEP(data_dict, fit_dict, resfit_dict, main_dict, meas_dict, V_KIDs, MB_filename, filenames, saveplots=False):
    N_pwr = data_dict['N_pwr']
    N_temp = data_dict['N_temp']
    # for i_pwr in range(N_pwr):
    #     for i_temp in range(N_temp):    
    #         fig = plt.figure(figsize=(14, 9))
    #         plt.subplots_adjust(bottom=0.09, top=0.94, right=0.6, left=0.08)
            
    #         T = data_dict['FP_temp'][i_temp][i_pwr]
    #         pwr = data_dict['pwr@device'][i_temp][i_pwr]
    #         fres = resfit_dict['fres'][i_temp][i_pwr]
    #         delta, alpha, f0 = get_MB_params(fres, T, MB_filename)
    #         nqp_th = func_nqp_nogap(delta, T)
    #         E = data_dict['E'][i_temp][i_pwr]
    #         # for i_pwr in range(N_pwr):
    #         tau_qp = fit_dict['tau_qp'][i_temp][i_pwr]
    #         Nqp = fit_dict['Nqp'][i_temp][i_pwr]
    #         # Cte = fit_dict['Cte'][i_temp][0]
            
    #         fres_cutoff = resfit_dict['fres_cutoff'][i_temp][i_pwr]
    #         SNqp_fit = SNqp_fit_func(data_dict['Sdff_f_clean'], tau_qp, Nqp, fres_cutoff)#, Cte)
    #         str_fit = r'GR Noise fit: $\tau_{qp}=%.0f \mu s$, $n_{qp}=%.2e \mu m^{-3}$'%((tau_qp)*1e6, Nqp/V_KIDs)
    #         str_fit += '\n$n_{qp, th}=%.2e \mu m^{-3}$'%(nqp_th)
    #         str_fit += '\n'
    #         plt.loglog(data_dict['Sdff_f_clean'], data_dict['SNqp_onres'][i_pwr][i_temp], 'oC%d'%i_temp, markersize=3, label = 'data with T = %d mK, pwr = %d dB, E= %f'%(T*1e3, pwr, E))
    #         plt.loglog(data_dict['Sdff_f_clean'], SNqp_fit, '-', linewidth=2,  color='C%d'%i_temp, label=str_fit)
    
    #         plt.xticks(color='k', size=18)
    #         plt.yticks(color='k', size=18)
    #         plt.xlabel("Frequency [Hz]", fontsize = 20)
    #         plt.ylabel(r'$S_{Nqp}$ [$Hz^{-1}$]', fontsize = 20)
    #         plt.legend(fontsize = 14)
    #         plt.legend(loc="upper right", bbox_to_anchor=[0.99, 0.94], bbox_transform = plt.gcf().transFigure, ncol=1, shadow=True, fancybox=True, fontsize=12)
    #         plt.grid(visible=True, which='both', color='0.65', linestyle='-')
    #         # plt.text(0.5, 0.97, meas_dict['str_title_res'], fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    #         # plt.text(0.5, 0.92, str_params1, fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    #         # plt.xlim([Sdff_dict['f'][2], Sdff_dict['f'][-1]])
    #         plt.ylim(1e-1, 1e4)
    #         if saveplots:
    #             file_dir = main_dict['plot_dir'] + 'SNqp/'
    #             os.makedirs(file_dir, exist_ok=True)
    #             filename = 'noiseSNqp_fitres_%sMHz_pwr_%ddBm_T_%dmK'%(meas_dict['freq_center_str'], pwr, int(T*1000))
    #             # if i_pwr == 0 and i_temp == 0:
    #             #     filenames=save_plots(main_dict['plot_dir'], [filename], [fig])
    #             #     fig.savefig(file_dir + filename  + '.png', dpi=mydpi, bbox_inches = 'tight')
    #             # else:
    #             fig.savefig(file_dir + filename  + '.png', dpi=mydpi, bbox_inches = 'tight')

        
    #         fig = plt.figure(figsize=(14, 9))
    #         plt.subplots_adjust(bottom=0.09, top=0.94, right=0.6, left=0.08)
    #         plt.loglog(data_dict['Sdff_f_clean'], data_dict['NEP'][i_pwr][i_temp], 'oC%d'%i_temp, markersize=3, label = 'data with T = %d mK, pwr = %d dB, E= %f'%(T*1e3, pwr, E))
    #         # plt.loglog(data_dict['Sdff_f_clean'], SNqp_fit, '-', linewidth=2,  color='C%d'%i_temp, label=str_fit)
        
    #         plt.xticks(color='k', size=18)
    #         plt.yticks(color='k', size=18)
    #         plt.xlabel("Frequency [Hz]", fontsize = 20)
    #         plt.ylabel(r'$NEP$ [$\sqrt{Hz}$]', fontsize = 20)
    #         plt.legend(fontsize = 14)
    #         plt.legend(loc="upper right", bbox_to_anchor=[0.99, 0.94], bbox_transform = plt.gcf().transFigure, ncol=1, shadow=True, fancybox=True, fontsize=12)
    #         plt.grid(visible=True, which='both', color='0.65', linestyle='-')
    #         # plt.text(0.5, 0.97, meas_dict['str_title_res'], fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    #         # plt.text(0.5, 0.92, str_params1, fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    #         # plt.xlim([Sdff_dict['f'][2], Sdff_dict['f'][-1]])
    #         plt.ylim(1e-18, 1e-14)
    #         if saveplots:
    #             file_dir = main_dict['plot_dir'] + 'NEP/'
    #             os.makedirs(file_dir, exist_ok=True)
    #             filename = 'noiseNEP_fitres_%sMHz_pwr_%ddBm_T_%dmK'%(meas_dict['freq_center_str'], pwr, int(T*1000))
    #             # if i_pwr == 0 and i_temp == 0:
    #             #     filenames=save_plots(main_dict['plot_dir'], [filename], [fig])
    #             #     fig.savefig(file_dir + filename  + '.png', dpi=mydpi, bbox_inches = 'tight')
    #             # else:
    #             fig.savefig(file_dir + filename  + '.png', dpi=mydpi, bbox_inches = 'tight')
                    
    # for i_pwr in range(N_pwr):
    i_pwr=0
    fig = plt.figure(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.09, top=0.94, right=0.6, left=0.08)
    for i_temp in range(N_temp):        
        T = data_dict['FP_temp'][i_temp][i_pwr]
        pwr = data_dict['pwr@device'][i_temp][i_pwr]
        fres = resfit_dict['fres'][i_temp][i_pwr]
        delta, alpha, f0 = get_MB_params(fres, T, MB_filename)
        nqp_th = func_nqp_nogap(delta, T)
        E = data_dict['E'][i_temp][i_pwr]
        # for i_pwr in range(N_pwr):
        tau_qp = fit_dict['tau_qp'][i_temp][i_pwr]
        Nqp = fit_dict['Nqp'][i_temp][i_pwr]
        # Cte = fit_dict['Cte'][i_temp][0]
        
        fres_cutoff = resfit_dict['fres_cutoff'][i_temp][i_pwr]
        SNqp_fit = SNqp_fit_func(data_dict['Sdff_f_clean'], tau_qp, Nqp, fres_cutoff)#, Cte)
        str_fit = r'GR Noise fit: $\tau_{qp}=%.0f \mu s$, $n_{qp}=%.2e \mu m^{-3}$'%((tau_qp)*1e6, Nqp/V_KIDs)
        str_fit += '\n$n_{qp, th}=%.2e \mu m^{-3}$'%(nqp_th)
        str_fit += '\n'
        plt.loglog(data_dict['Sdff_f_clean'], data_dict['SNqp_onres'][i_pwr][i_temp], 'oC%d'%i_temp, markersize=3, label = 'data with T = %d mK, pwr = %d dB, E= %f'%(T*1e3, pwr, E))
        plt.loglog(data_dict['Sdff_f_clean'], SNqp_fit, '-', linewidth=2,  color='C%d'%i_temp, label=str_fit)

    plt.xticks(color='k', size=18)
    plt.yticks(color='k', size=18)
    plt.xlabel("Frequency [Hz]", fontsize = 20)
    plt.ylabel(r'$S_{Nqp}$ [$Hz^{-1}$]', fontsize = 20)
    plt.title('noise SNqp sweepT:  fitres %sMHz  pwr %ddBm'%(meas_dict['freq_center_str'], pwr))
    plt.legend(fontsize = 14)
    plt.legend(loc="upper right", bbox_to_anchor=[0.99, 0.94], bbox_transform = plt.gcf().transFigure, ncol=1, shadow=True, fancybox=True, fontsize=12)
    plt.grid(visible=True, which='both', color='0.65', linestyle='-')
    # plt.text(0.5, 0.97, meas_dict['str_title_res'], fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    # plt.text(0.5, 0.92, str_params1, fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    # plt.xlim([Sdff_dict['f'][2], Sdff_dict['f'][-1]])
    plt.ylim(1e-1, 1e4)
    if saveplots:
        filename = ['noiseSNqp_sweepT_fitres_%sMHz_pwr_%ddBm'%(meas_dict['freq_center_str'], pwr)]
        filenames.append(save_plots(main_dict['plot_dir'] + 'res' + str(main_dict['res_ID']) + '/', filename, [fig]))
    
    fig = plt.figure(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.09, top=0.94, right=0.6, left=0.08)
    # i_temp=0
    # for i_pwr in range(N_pwr):
    # i_pwr=i_dex
    for i_temp in range(N_temp):
        T = data_dict['FP_temp'][i_temp][i_pwr]
        pwr = data_dict['pwr@device'][i_temp][i_pwr]
        E = data_dict['E'][i_temp][i_pwr]
        plt.loglog(data_dict['Sdff_f_clean'], data_dict['NEP'][i_pwr][i_temp], 'oC%d'%i_temp, markersize=3, label = 'data with T = %d mK, pwr = %d dB, E= %f'%(T*1e3, pwr, E))
        # plt.loglog(data_dict['Sdff_f_clean'], SNqp_fit, '-', linewidth=2,  color='C%d'%i_temp, label=str_fit)

    plt.xticks(color='k', size=18)
    plt.yticks(color='k', size=18)
    plt.xlabel("Frequency [Hz]", fontsize = 20)
    plt.ylabel(r'$NEP$ [$\sqrt{Hz}$]', fontsize = 20)
    plt.title('noise NEP sweepT:  fitres %sMHz  pwr %ddBm'%(meas_dict['freq_center_str'], pwr))
    plt.legend(fontsize = 14)
    plt.legend(loc="upper right", bbox_to_anchor=[0.99, 0.94], bbox_transform = plt.gcf().transFigure, ncol=1, shadow=True, fancybox=True, fontsize=12)
    plt.grid(visible=True, which='both', color='0.65', linestyle='-')
    # plt.text(0.5, 0.97, meas_dict['str_title_res'], fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    # plt.text(0.5, 0.92, str_params1, fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    # plt.xlim([Sdff_dict['f'][2], Sdff_dict['f'][-1]])
    plt.ylim(1e-18, 1e-14)
    if saveplots:
        filename = ['noiseNEP_sweepT_fitres_%sMHz_pwr_%ddBm'%(meas_dict['freq_center_str'], pwr)]
        filenames.append(save_plots(main_dict['plot_dir'] + 'res' + str(main_dict['res_ID']) + '/', filename, [fig]))

    i_temp=0    
    # for i_temp in range(N_temp): 
    fig = plt.figure(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.09, top=0.94, right=0.6, left=0.08)
    for i_pwr in range(N_pwr):       
        T = data_dict['FP_temp'][i_temp][i_pwr]
        pwr = data_dict['pwr@device'][i_temp][i_pwr]
        fres = resfit_dict['fres'][i_temp][i_pwr]
        delta, alpha, f0 = get_MB_params(fres, T, MB_filename)
        nqp_th = func_nqp_nogap(delta, T)
        E = data_dict['E'][i_temp][i_pwr]
        # for i_pwr in range(N_pwr):
        tau_qp = fit_dict['tau_qp'][i_temp][i_pwr]
        Nqp = fit_dict['Nqp'][i_temp][i_pwr]
        # Cte = fit_dict['Cte'][i_temp][0]
        
        fres_cutoff = resfit_dict['fres_cutoff'][i_temp][i_pwr]
        SNqp_fit = SNqp_fit_func(data_dict['Sdff_f_clean'], tau_qp, Nqp, fres_cutoff)#, Cte)
        str_fit = r'GR Noise fit: $\tau_{qp}=%.0f \mu s$, $n_{qp}=%.2e \mu m^{-3}$'%((tau_qp)*1e6, Nqp/V_KIDs)
        str_fit += '\n$n_{qp, th}=%.2e \mu m^{-3}$'%(nqp_th)
        str_fit += '\n'
        plt.loglog(data_dict['Sdff_f_clean'], data_dict['SNqp_onres'][i_pwr][i_temp], 'oC%d'%i_pwr, markersize=3, label = 'data with T = %d mK, pwr = %d dB, E= %f'%(T*1e3, pwr, E))
        plt.loglog(data_dict['Sdff_f_clean'], SNqp_fit, '-', linewidth=2,  color='C%d'%i_pwr, label=str_fit)

    plt.xticks(color='k', size=18)
    plt.yticks(color='k', size=18)
    plt.xlabel("Frequency [Hz]", fontsize = 20)
    plt.ylabel(r'$S_{Nqp}$ [$Hz^{-1}$]', fontsize = 20)
    plt.title('noise SNqp sweep Power:  fitres %sMHz  T %dmK'%(meas_dict['freq_center_str'], int(T*1000)))
    plt.legend(fontsize = 14)
    plt.legend(loc="upper right", bbox_to_anchor=[0.99, 0.94], bbox_transform = plt.gcf().transFigure, ncol=1, shadow=True, fancybox=True, fontsize=12)
    plt.grid(visible=True, which='both', color='0.65', linestyle='-')
    # plt.text(0.5, 0.97, meas_dict['str_title_res'], fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    # plt.text(0.5, 0.92, str_params1, fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    # plt.xlim([Sdff_dict['f'][2], Sdff_dict['f'][-1]])
    plt.ylim(1e-1, 1e4)
    if saveplots:
        filename = ['noiseSNqp_sweepP_fitres_%sMHz_T_%dmK'%(meas_dict['freq_center_str'], int(T*1000))]
        filenames.append(save_plots(main_dict['plot_dir'] + 'res' + str(main_dict['res_ID']) + '/', filename, [fig]))
    
    fig = plt.figure(figsize=(14, 9))
    plt.subplots_adjust(bottom=0.09, top=0.94, right=0.6, left=0.08)

    for i_pwr in range(N_pwr):
        T = data_dict['FP_temp'][i_temp][i_pwr]
        pwr = data_dict['pwr@device'][i_temp][i_pwr]
        E = data_dict['E'][i_temp][i_pwr]
        plt.loglog(data_dict['Sdff_f_clean'], data_dict['NEP'][i_pwr][i_temp], 'oC%d'%i_pwr, markersize=3, label = 'data with T = %d mK, pwr = %d dB, E= %f'%(T*1e3, pwr, E))
        # plt.loglog(data_dict['Sdff_f_clean'], SNqp_fit, '-', linewidth=2,  color='C%d'%i_temp, label=str_fit)

    plt.xticks(color='k', size=18)
    plt.yticks(color='k', size=18)
    plt.xlabel("Frequency [Hz]", fontsize = 20)
    plt.ylabel(r'$NEP$ [$\sqrt{Hz}$]', fontsize = 20)
    plt.title('noise NEP sweep Power:  fitres %sMHz  T %dmK'%(meas_dict['freq_center_str'], int(T*1000)))
    plt.legend(fontsize = 14)
    plt.legend(loc="upper right", bbox_to_anchor=[0.99, 0.94], bbox_transform = plt.gcf().transFigure, ncol=1, shadow=True, fancybox=True, fontsize=12)
    plt.grid(visible=True, which='both', color='0.65', linestyle='-')
    # plt.text(0.5, 0.97, meas_dict['str_title_res'], fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    # plt.text(0.5, 0.92, str_params1, fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    # plt.xlim([Sdff_dict['f'][2], Sdff_dict['f'][-1]])
    plt.ylim(1e-18, 1e-14)
    if saveplots:
        filename = ['noiseNEP_sweepP_fitres_%sMHz_T_%dmK'%(meas_dict['freq_center_str'], int(T*1000))]
        filenames.append(save_plots(main_dict['plot_dir'] + 'res' + str(main_dict['res_ID']) + '/', filename, [fig]))


    plt.close('all')
    return filenames

    


# SNqp = func_SNqp(fres, T, Sdff_freq_mean, delta, V_KIDs)

# fit_dict = fit_tau_Nqp(f_Sdff_trunc, SNqp, f_range=(2e1, 5e4))




xi = lambda omega, T: (hbar_eV  * omega)/(2* kb_eV * T)


# k2 calculation
def get_k2(omega, T, delta_0, N0):
    # delta = get_delta(delta_0, T)
    k2 = 1/(2*N0*delta_0) * (1 + np.sqrt(2*delta_0/(np.pi*kb_eV*T)) * np.exp(-xi(omega, T)) * sp.iv(0, xi(omega, T)))
    return k2


def func_SNqp(fres, T, Sdff_freq, delta_0, alpha, V):
    # k2 = get_k2(Sdff_dict['Sdff_onres_freq']*2*np.pi, T*1e-3, delta_0, N0)
    # SNqp = 4 * V**2 * Sdff_dict['Sdff_onres_freq'] / k2**2 
    k2 = get_k2(fres*2*np.pi, T, delta_0, N0)
    SNqp = 4 * V**2 * Sdff_freq / ( alpha**2 * k2**2 )
    return SNqp
    

def SNqp_fit_func(f, tau_qp, Nqp, fres_cutoff):#, Cte):
    reso_filt = np.abs(1 / (1 + 1j * f / fres_cutoff))**2
    qp_filt = (4 * tau_qp * Nqp) / (1 + (tau_qp * 2*np.pi*f)**2)
    return qp_filt * reso_filt #+ params['Cte']


def func_nqp_nogap(delta, T, mu=0): 
    return 4 * N0 * delta * np.exp((mu)/(kb_eV * T)) * sp.kn(1, delta/(kb_eV*T))


def get_MB_params(freq, T, MB_filename):
    Delta = []
    alpha = []
    f0 = []
    nqp = []
    print(' --- Read h5 MB fit file ---')
    with h5py.File(MB_filename, 'r') as hf: 
        # hf = h5py.File(MB_filename, 'r')
        key_source = list(hf.keys())[0]
        gr_MB_fit = hf[key_source]['MB_fit']
        for res in gr_MB_fit.keys():
            Delta.append(gr_MB_fit[res]['Fres_fit']['Delta'][()])
            alpha.append(gr_MB_fit[res]['Fres_fit']['alpha'][()])
            f0.append(gr_MB_fit[res]['Fres_fit']['f0'][()])
            nqp.append(gr_MB_fit[res]['Fres_fit']['nqp'][()])

    Delta = np.array(Delta)
    alpha = np.array(alpha)
    f0 = np.array(f0)
    nqp = np.array(nqp)
    # for i in range(len(Delta)):
    omega = 2*np.pi*freq
    k2 = get_k2(omega, T, Delta, N0)
    res_i = np.argmin(np.abs(-alpha/2 * k2 * func_nqp_nogap(Delta, T) - (freq - f0) / f0))
    return Delta[res_i], alpha[res_i], f0[res_i]


def get_E(data_dict, resfit_dict):
    N_pwr = data_dict['N_pwr']
    N_temp = data_dict['N_temp']
    Pa=10**21/(11.68*8.854187817*0.000000375*800)
    data_dict['E'] = np.array(np.zeros([N_temp, N_pwr]))
    for i_temp in range(N_temp):
        for i_pwr in range(N_pwr):
            pwr = data_dict['pwr@device'][i_temp][i_pwr]
            fres=resfit_dict['fres'][i_temp][i_pwr]
            Qr=resfit_dict['Qr'][i_temp][i_pwr]
            Qc=resfit_dict['Qc'][i_temp][i_pwr]
            data_dict['E'][i_temp][i_pwr]=np.sqrt(Pa)*np.sqrt(10**(pwr/10-3)*Qr**2/(2*np.pi*fres*Qc))
            
    return data_dict
  
def vrms2dbm(vp):
    """
    Converts a scalar or a numpy array from volts RMS to dbm assuming there is an impedence of 50 Ohm
    Arguments:
        - vp: scalar or numpy array containig values in volt RMS to be converted in dmb
    Returns:
        - scalar or numpy array containing the result
    """
    return 10 * np.log10(20 * vp**2)

    
def save_plots(file_path, filename, figs):
    file_path = file_path + 'report/'
    os.makedirs(file_path, exist_ok=True)
    # print('Saving %s at %s MHz with %s dBm pwr'%(filedir_name, meas_dict['freq_center'], meas_dict['pwr@device']))
    for i,fig in enumerate(figs):
        # plt.savefig(file_path + file_name  + '.svg', dpi=mydpi, bbox_inches = 'tight')
        plt.savefig(file_path + filename[i]  + '.png', dpi=mydpi, bbox_inches = 'tight')
        # plt.savefig(file_path + file_name  + '_tiny.png', dpi=int(mydpi/4.5), bbox_inches = 'tight')
        plt.savefig(file_path + filename[i]  + '.pdf', dpi=mydpi, bbox_inches = 'tight')
        filename[i]=file_path + filename[i]  + '.pdf'
    
    return filename

    
    
def save_pdf_report(file_path, meas_dict, filenames):
    filenames_flat = np.hstack(filenames)
    pdf_merger = PdfFileMerger()
    # merger_filename = dev_name + '_%ddBm_%dmK_report.pdf'%(pwrs[0], temps[i])
    merger_filename = 'res%s_report.pdf'%(meas_dict['freq_center_str'])
    for file in filenames_flat:
        pdf_merger.append(file)
    if os.path.exists(file_path + 'report/' + merger_filename):
        os.remove(file_path + 'report/' + merger_filename)
    pdf_merger.write(file_path + 'report/' + merger_filename)
    pdf_merger.close() 
    
    
def KsNormDetect(data):
    u = np.mean(data)
    std = np.std(data)
    res = kstest(data, 'norm', (u, std))[1]
    out=[]
    if res<=0.05:
        print('the data are from a normal distribution')
        out1 = np.where(((u - 3*std) > data))[0]
        out2 = np.where(((u + 3*std) < data))[0]
        out = np.concatenate([out1, out2])
    else:
        print('no selected')
    
    return out
  
def save_data(main_dict, data_dict, fit_dict):
    print('--- Saving data ---')
    newfile_path = main_dict['processed_dir']
    if os.path.isfile(newfile_path):
        os.remove(newfile_path)
    print("Copying the original file as new processed file")
    copy(main_dict['original_dir'], newfile_path)
    print('Saving the processed data...')
    with h5py.File(newfile_path, 'a') as hf: 
        key_BBtemp = list(hf.keys())[0]
        for i_tmp, key_FPtemp in enumerate(hf[key_BBtemp].keys()):
            for i_pwr, key_power in enumerate(hf[key_BBtemp][key_FPtemp].keys()):
                gr_res = hf[key_BBtemp][key_FPtemp][key_power]['res_scan']
                gr_Sdff = gr_res.create_group('Sdff_clean')
                gr_Sdff.create_dataset('f', data=data_dict['f'][i_tmp][i_pwr])
                gr_Sdff.create_dataset('Sdff_onres_freq_clean', data=data_dict['Sdff_onres_freq_clean'][i_tmp][i_pwr])
                gr_Sdff.create_dataset('Sdff_onres_diss_clean', data=data_dict['Sdff_onres_diss_clean'][i_tmp][i_pwr])
                gr_Sdff.create_dataset('Sdff_offres_freq_clean', data=data_dict['Sdff_offres_freq_clean'][i_tmp][i_pwr])
                gr_Sdff.create_dataset('Sdff_offres_diss_clean', data=data_dict['Sdff_offres_diss_clean'][i_tmp][i_pwr])
                gr_Sdff.create_dataset('noise_select_index', data=data_dict['noise_select_index'][i_tmp][i_pwr])
                
                gr_SNqp = hf[key_BBtemp][key_FPtemp][key_power]['res_scan'].create_group('SNqp')
                f_dataset = gr_SNqp.create_dataset('f', data=data_dict['f'][i_tmp][i_pwr])
                SNqp_dataset = gr_SNqp.create_dataset('SNqp', data=data_dict['SNqp_onres'][i_tmp][i_pwr])
                SNqp_fit_dataset = gr_SNqp.create_dataset('SNqp_fit', data=data_dict['SNqp_fit'][i_tmp][i_pwr])
                tau_qp_dataset = gr_SNqp.create_dataset('tau_qp', data=fit_dict['tau_qp'][i_tmp][i_pwr])
                Nqp_dataset = gr_SNqp.create_dataset('Nqp', data=fit_dict['Nqp'][i_tmp][i_pwr])
                nqp_th_dataset = gr_SNqp.create_dataset('nqp_th', data=data_dict['nqp_th'][i_tmp][i_pwr])
                f_dataset.attrs['unit'] = 'Hz'
                SNqp_dataset.attrs['unit'] = '/Hz'
                
                gr_NEP = hf[key_BBtemp][key_FPtemp][key_power]['res_scan'].create_group('NEP')
                f_dataset = gr_NEP.create_dataset('f', data=data_dict['f'][i_tmp][i_pwr])
                NEP_dataset = gr_NEP.create_dataset('NEP', data=data_dict['NEP'][i_tmp][i_pwr])
                E_dataset = gr_NEP.create_dataset('E', data=data_dict['E'][i_tmp][i_pwr])
                Pa_dataset = gr_NEP.create_dataset('Pa', data=data_dict['Pa'])
                eta_pb_dataset = gr_NEP.create_dataset('eta_pb', data=data_dict['eta_pb'])
                f_dataset.attrs['unit'] = 'Hz'
                NEP_dataset.attrs['unit'] = 'W/sqrt{Hz}'

def check_data(dir_path):
    root = tk.Tk()
    file_path = filedialog.askopenfilename(title='Choose .h5 file with Sdff data', initialdir=dir_path)
    filename = file_path.split('/')[:-3]
    root.destroy()

    with h5py.File(file_path, 'r') as f:
        key_BBtemp = list(f.keys())[0]
        key_temp = list(f[key_BBtemp].keys())[0]
        key_pwr = list(f[key_BBtemp][key_temp].keys())[0]
        
        # f_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['freq_data'][()]
        I_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['fit']['I_data'][()]
        Q_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['fit']['Q_data'][()]
        
        Amp_data = np.sqrt(np.square(I_data)+np.square(Q_data))
        Ang_data = phase_scale(I_data, Q_data)
    
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9))
        axs[0].plot(I_data, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)
        axs[0].set_ylabel('I [V]', fontsize = 12)
        axs[1].plot(Q_data, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)
        axs[1].set_ylabel('Q [V]', fontsize = 12)
        axs[2].plot(Amp_data, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)
        axs[2].set_ylabel('radius [V]', fontsize = 10)
        axs[3].plot(Ang_data, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)
        axs[3].set_ylabel('Phase [rad]', fontsize = 10)
        fig.legend()
        
def check_twotonedata(dir_path):
    root = tk.Tk()
    file_path = filedialog.askopenfilename(title='Choose .h5 file with Sdff data', initialdir=dir_path)
    filename = file_path.split('/')[:-3]
    root.destroy()

    with h5py.File(file_path, 'r') as f:
        key_BBtemp = list(f.keys())[0]
        key_temp = list(f[key_BBtemp].keys())[0]
        key_pwr = list(f[key_BBtemp][key_temp].keys())[0]
        
        m_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['WaveFormG'][()]
        # I_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['fit']['I_data'][()]
        # Q_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['fit']['Q_data'][()]
        
        # Amp_data = np.sqrt(np.square(I_data)+np.square(Q_data))
        # Ang_data = phase_scale(I_data, Q_data)
    
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(16, 9))
        axs[0].plot(m_data, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)
        axs[0].set_ylabel('I [V]', fontsize = 12)
        # axs[1].plot(Q_data, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)
        # axs[1].set_ylabel('Q [V]', fontsize = 12)
        # axs[2].plot(Amp_data, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)
        # axs[2].set_ylabel('radius [V]', fontsize = 10)
        # axs[3].plot(Ang_data, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)
        # axs[3].set_ylabel('Phase [rad]', fontsize = 10)
        fig.legend()
        S_time=1
        fI, AmpI, AngI = FFT(m_data,S_time)
        peaks, _ = sig.find_peaks(np.abs(AmpI), prominence=0.1, distance=200e3*S_time)
        print(peaks)

def get_allresFreq_data(dir_path):
    root = tk.Tk()
    file_path = filedialog.askopenfilename(title='Choose .h5 file with Sdff data', initialdir=dir_path)
    filename = file_path.split('/')[:-3]
    root.destroy()

    with h5py.File(file_path, 'r') as f:
        key_load = list(f.keys())[0]
        key_temp = list(f[key_load].keys())[1]
        key_pwr = list(f[key_load][key_temp].keys())[0]
    
        ResList = list(f[key_load][key_temp][key_pwr]['split'].keys())
        esNumber = len(ResList)
        ResFreq=np.zeros(esNumber)
        for rr, Res in enumerate(ResList):
            ResFreq[rr] = f[key_load][key_temp][key_pwr]['split'][Res].attrs['res']
        print(ResFreq)
    return ResFreq

def FFT(data, Time, firstAmp_half=True):
    L = len(data)
    fft_y = fft(data)
    axisFreq = np.arange(int(L/2)) / Time
    Ampresult = np.abs(fft_y) / L * 2
    Ampresult = Ampresult[range(int(L/2))]
    if firstAmp_half:
        Ampresult[0] /= 2
    Angresult = np.angle(fft_y)
    Angresult = Angresult[range(int(L/2))]
    return axisFreq, Ampresult, Angresult

def butter_lowpass(cutoff, fs, order=5, plots=None):
    """
    Design a Butterworth lowpass filter.
    
    Parameters:
    - cutoff (float): Cutoff frequency in Hz.
    - fs (float): Sampling frequency in Hz.
    - order (int): The order of the filter (default is 5).
    - plots (str, display or None): Path to save the frequency response plot, 'display' to only display the plot or None to skip plotting [default].
    
    Returns:
    - b, a (ndarray): Numerator (b) and denominator (a) filter coefficients.
    
    Use the following function to apply the filter: butter_lowpass_filter(data, cutoff, fs, order)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = sig.butter(order, normal_cutoff, btype='low', analog=False, output = 'sos')
    
    #if plots:
    #    plot_filter_response(b, a, fs, f"Lowpass Filter (cutoff = {cutoff} Hz)", plots)
        
    return sos

def butter_lowpass_filter(data, cutoff, fs, order=5, plots=None):
    """
    Apply a lowpass Butterworth filter to a data sequence.
    
    Parameters:
    - data (ndarray): Input data to filter.
    - cutoff (float): Cutoff frequency in Hz.
    - fs (float): Sampling frequency in Hz.
    - order (int): The order of the filter (default is 5).
    - plots (str, display or None): Path to save the frequency response plot, 'display' to only display the plot or None to skip plotting [default].
    
    Returns:
    - y (ndarray): Filtered data.
    """
    sos= butter_lowpass(cutoff, fs, order=order, plots=plots)
    y = sig.sosfilt(sos, data)
    return y

def butter_highpass(cutoff, fs, order=5, plots=None):
    """
    Design a Butterworth highpass filter.
    
    Parameters:
    - cutoff (float): Cutoff frequency in Hz.
    - fs (float): Sampling frequency in Hz.
    - order (int): The order of the filter (default is 5).
    - plots (str, display or None): Path to save the frequency response plot, 'display' to only display the plot or None to skip plotting [default].
    
    Returns:
    - b, a (ndarray): Numerator (b) and denominator (a) filter coefficients.
    
    Use the following function to apply the filter: butter_highpass_filter(data, cutoff, fs, order)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype="high", analog=False)
    
    if plots:
        plot_filter_response(b, a, fs, f"Highpass Filter (cutoff = {cutoff} Hz)", plots)
        
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5, plots=None):
    """
    Apply a highpass Butterworth filter to a data sequence.
    
    Parameters:
    - data (ndarray): Input data to filter.
    - cutoff (float): Cutoff frequency in Hz.
    - fs (float): Sampling frequency in Hz.
    - order (int): The order of the filter (default is 5).
    - plots (str, display or None): Path to save the frequency response plot, 'display' to only display the plot or None to skip plotting [default].
    
    Returns:
    - y (ndarray): Filtered data.
    """
    b, a = butter_highpass(cutoff, fs, order=order, plots=plots)
    y = sig.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5, plots=None):
    """
    Design a Butterworth bandpass filter.
    
    Parameters:
    - lowcut (float): Lower cutoff frequency in Hz.
    - highcut (float): Upper cutoff frequency in Hz.
    - fs (float): Sampling frequency in Hz.
    - order (int): The order of the filter (default is 5).
    - plots (str, display or None): Path to save the frequency response plot, 'display' to only display the plot or None to skip plotting [default].
    
    Returns:
    - b, a (ndarray): Numerator (b) and denominator (a) filter coefficients.
    
    Use the following function to apply the filter: butter_bandpass_filter(data, lowcut, highcut, fs, order)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    
    if plots:
        plot_filter_response(b, a, fs, f"Bandpass Filter (lowcut = {lowcut} Hz, highcut = {highcut} Hz)", plots)
        
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, plots=None):
    """
    Apply a bandpass Butterworth filter to a data sequence.
    
    Parameters:
    - data (ndarray): Input data to filter.
    - lowcut (float): Lower cutoff frequency in Hz.
    - highcut (float): Upper cutoff frequency in Hz.
    - fs (float): Sampling frequency in Hz.
    - order (int): The order of the filter (default is 5).
    - plots (str, display or None): Path to save the frequency response plot, 'display' to only display the plot or None to skip plotting [default].
    
    Returns:
    - y (ndarray): Filtered data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, plots=plots)
    y = sig.lfilter(b, a, data)
    # y = sig.filtfilt(b, a, data)
    return y

def plot_filter_response(b, a, fs, title, plot_option="display"):
    """
    Plot and either display or save the frequency response of a filter.
    
    Parameters:
    - b, a (ndarray): Numerator (b) and denominator (a) filter coefficients.
    - fs (float): Sampling frequency in Hz.
    - title (str): Title of the plot for display purposes.
    - plot_option (str): If "display", displays the plot. If a valid file path is provided, 
                         saves the plot at that path. Defaults to "display".
                         If the directory does not exist, plot will be displayed instead.
    """
    w, h = sig.freqz(b, a, worN=8000)
    freqs = w * fs / (2 * np.pi)  # Convert to Hz
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 20 * np.log10(abs(h)), 'b')
    plt.title(f'{title} Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid()

    if plot_option == "display" or not os.path.isdir(os.path.dirname(plot_option)):
        plt.show()
    else:
        plt.savefig(plot_option)
        plt.close()


def numerical_demodulation(I_data,
                           Q_data,
                           fs,
                           CarrierTones,
                           BandWidth,
                           Filter_order=10,
                           WaveForm_data = None,
                           PhaseCorrection = False,
                           Throw = False,
                           plots = True,
                           Plots_dir = None,
                           Noise_debug = False,
                           Special_Sweep = False
                           ):
    """
    Perform numerical demodulation of I and Q data.

    Parameters:
    - I_data, Q_data: Input I and Q data arrays.
    - fs (float): Sampling frequency in Hz.
    - CarrierTones (array): Array of carrier tone frequencies.
    - BandWidth (float): Bandwidth for filtering in Hz.
    - WaveForm_data (array or None): Optional waveform data array.
    - PhaseCorrection (bool): If True, apply phase correction.
    - plots (bool): If True [default], enables plotting with optional path for saving plots.
    - Plots_dir (str or None): Directory path to save plots if specified, if None [default and plots is True [default], only display the plots

    Returns:
    - I_demodulated (list): Demodulated I data.
    - Q_demodulated (list): Demodulated Q data.
    """
    
    print('Starting numerical demodulation')
    
    
    # debug = True
    debug = False
    
    # Noise_debug = False
    # Noise_debug = True

    
    assert WaveForm_data is not None, "I didn't code this part yet"
    # assert PhaseCorrection == False, "I didn't code this part yet"
    
    assert len(I_data.shape) == 1 and len(Q_data.shape) == 1, 'Wrong data format for I and Q data, please give a 1D array.'
    assert len(I_data) == len(Q_data), "I and Q don't have the same size"
    if WaveForm_data is not None:
        assert len(WaveForm_data.shape) == 1, 'Wrong data format for the WaveForm signal, please give a 1D array.'
        assert len(WaveForm_data) == len(Q_data), "The WaveForm data don't have the same size than I and Q"
    
    assert CarrierTones[0]>BandWidth, f'The frequency distance between the first carrier tone to DC is smaller than the BandWidth of {BandWidth} Hz'
    if CarrierTones.shape[0]>1:
        for ii,Tone in enumerate(CarrierTones[1:]):
            assert Tone-CarrierTones[ii] >= 2*BandWidth, \
                f'The frequency distance between the carriers is below the targeted bandwidth of {BandWidth} Hz'
       
    Npts = I_data.shape[0]
    S_time = (Npts) / fs
    timedata = np.linspace(0, S_time, Npts, endpoint=False)
    
    # Initialize plot_range to full range by default
    plot_range = slice(None)
    
    if Throw == False:
        throw = 0
        S_time_throw = S_time
    else:
        throw = int(0.1*Npts)
        S_time_throw = len(I_data[throw:])/fs
    
    # Adjust plot_range based on conditions
    if Throw:
        throw = int(0.1*Npts)
        plot_range = slice(Throw, None)
    elif Noise_debug:
        Npts_reduced = int(Npts * 0.01 / 100)
        plot_range = slice(Npts//2, Npts//2+Npts_reduced)
    

    if WaveForm_data is not None:
        # Treating the WaveForm signal
        fWF, AmpWF, AngWF = FFT(WaveForm_data, S_time)
        WaveForm_data -= np.mean(WaveForm_data)
        # WaveForm_dataQ = phase_shift_signal(WaveForm_data,fs)
        # Find peaks for demodulation #I should introduce a check that the measured frequency match my carrier tones under a specific tolerance
        peaks2, _ = sig.find_peaks(AmpWF, prominence=0.01, distance= 2*BandWidth * S_time)
        print('Peaks found in waveform raw data', peaks2, 'corresponding to', (fWF[peaks2] / 1e3), 'kHz')
        if PhaseCorrection is False:
            AngWF = []
        # Store for multiple carriers
        WaveForm_filtered = []
        # WaveForm_filteredQ = []
        WF_phase_filt = []
        
        
        for tt in range(CarrierTones.shape[0]):
            
            # Normalize the waveform
            # WaveForm_data = WaveForm_data / AmpWF[peaks2[tt]]
            # AmpWF = AmpWF/AmpWF[peaks2[tt]]
            # Bandpass filtering based on carrier tone frequency
            WaveForm_filtered_data = butter_bandpass_filter(WaveForm_data, CarrierTones[tt] - BandWidth, CarrierTones[tt] + BandWidth, fs, Filter_order)#, plots)
            # WaveForm_filtered_dataQ = butter_bandpass_filter(WaveForm_dataQ, CarrierTones[tt] - BandWidth, CarrierTones[tt] + BandWidth, fs, Filter_order)#, plots)
            # WaveForm_filteredQ.append(WaveForm_filtered_dataQ[throw:])
            # FFT on filtered data and store individual results in lists
            fWF_filt, AmpWF_filt, AngWF_filt = FFT(WaveForm_filtered_data[throw:], S_time_throw)
            peaksWF, _ = sig.find_peaks(AmpWF_filt, prominence=0.01, distance= 2*BandWidth * S_time_throw)
            print('Peaks found in waveform filtered data', peaksWF, 'corresponding to', (fWF_filt[peaksWF] / 1e3), 'kHz')
            print(f'Normalization of the peak at {fWF_filt[peaksWF] / 1e3} kHz by the amplitude {AmpWF_filt[peaksWF[0]]} at the tone')
            # WaveForm_filtered_data /= AmpWF_filt[peaksWF[0]]
            # AmpWF_filt /= AmpWF_filt[peaksWF[0]]
            WaveForm_filtered_data = normalize_rms_amplitude(WaveForm_filtered_data)
            WaveForm_filtered.append(WaveForm_filtered_data[throw:])
            if PhaseCorrection is False:
                WF_phase_filt.append(AngWF_filt[peaksWF[0]])
                AngWF_filt = []
            else:
                WF_phase_filt.append(AngWF_filt[peaksWF[0]])
                # WF_phase_filt.append(AngWF[peaks2[0]])

            if plots is True:
                figWF = control_plots_demodulation(timedata[plot_range], [WaveForm_data[plot_range], WaveForm_filtered[tt][plot_range]], [fWF[plot_range], fWF_filt[plot_range]],\
                                                   [AmpWF[plot_range], AmpWF_filt[plot_range]], CarrierTones[tt], data_type = 'WaveForm', save_path = Plots_dir)
                # fWF_filt = []
                AmpWF_filt = []
            
        # Treating the I&Q signal
        fI, AmpI, AngI = FFT(I_data, S_time)
        fQ, AmpQ, AngQ = FFT(Q_data, S_time)
        if PhaseCorrection is False:
            AngI = []
            AngQ = []
        I_data -= np.mean(I_data)
        Q_data -= np.mean(Q_data)
        # Find peaks for demodulation
        peaks, _ = sig.find_peaks(np.abs(AmpI + 1j * AmpQ), prominence=0.01, distance= 2*BandWidth* S_time)
        print('Peaks found in I and Q raw data', peaks, 'corresponding to', (fI[peaks] / 1e3), 'kHz')
        # Store for multiple carriers
        I_demodulated_list = []
        Q_demodulated_list = []
        
        Delta_Phase_I_list = []
        # I signal
        print('Demodulating I')
        for tt in range(CarrierTones.shape[0]):
            
            # Bandpass filtering based on carrier tone frequency
            #if Special_Sweep:
            #    I_filtered_data = butter_lowpass_filter(I_data, CarrierTones[tt] + BandWidth, fs, Filter_order)#, plots)
            #else:
            I_filtered_data = butter_bandpass_filter(I_data, CarrierTones[tt] - BandWidth, CarrierTones[tt] + BandWidth, fs, Filter_order)#, plots)
            # Q_filtered_data = butter_bandpass_filter(Q_data, CarrierTones[tt] - BandWidth, CarrierTones[tt] + BandWidth, fs, Filter_order)#, plots)
            # FFT on filtered data and store individual results in lists
            fI_filt, AmpI_filt, AngI_filt = FFT(I_filtered_data[throw:], S_time_throw)
            # fQ_filt, AmpQ_filt, AngQ_filt = FFT(Q_filtered_data[throw:], S_time_throw)
            peaksI, _ = sig.find_peaks(np.abs(AmpI_filt), prominence=0.01, distance= 2*BandWidth* S_time_throw)
            # peaksI, _ = sig.find_peaks(np.abs(AmpI_filt+1j*AmpQ_filt), prominence=0.01, distance= 2*BandWidth* S_time_throw)
            print('Peaks found in I filtered data', peaksI, 'corresponding to', (fI_filt[peaksI] / 1e3), 'kHz')
            if PhaseCorrection is False:
                AngI_filtered = AngI_filt[peaksI[0]]
                Delta_Phase_I = -(WF_phase_filt[tt]-AngI_filtered)
                AngI_filt = []
                print(f'The phase difference between the WaveForm and I at the carrier tone {CarrierTones[tt]} is {Delta_Phase_I} rad')
                # Compute demodulated I
                I_demodulated_data = butter_lowpass_filter(2*WaveForm_filtered[tt][:] * I_filtered_data[throw:], BandWidth, fs, Filter_order)
                # I_demodulated_data = butter_lowpass_filter(-WaveForm_filtered[tt][:] * I_filtered_data[throw:], BandWidth, fs, Filter_order) #should not work
                print('Phase Correction is not activated')
            else:
                AngI_filtered = AngI_filt[peaksI[0]]
                # AngI_filtered = AngI[peaks[0]]
                Delta_Phase_I_list.append( -(WF_phase_filt[tt]-AngI_filtered))#-np.pi/2)
                AngI_filt = []
                print(f'The phase difference between the WaveForm and I at the carrier tone {CarrierTones[tt]} is {Delta_Phase_I_list[tt]} rad')
                # Compute demodulated I
                I_demodulated_data = \
                    butter_lowpass_filter(2*phase_shift_signal(WaveForm_filtered[tt][:], fs, shift=Delta_Phase_I_list[tt]/( 2*np.pi*fI_filt[peaksI[0]] ))* I_filtered_data[throw:],\
                                          BandWidth, fs, Filter_order)
                print('Phase Correction is activated and applied in demodulation.')
            I_demodulated_list.append(I_demodulated_data)
            if plots is True: 
                
                fI_demodulated, AmpI_demodulated, AngI_demodulated = FFT(I_demodulated_data[throw:], len(I_demodulated_data[throw:])/fs)
                figI = control_plots_demodulation(timedata[plot_range], [I_data[plot_range], I_filtered_data[plot_range], I_demodulated_data[plot_range]],\
                                                  [fI[plot_range], fI_filt[plot_range], fI_demodulated[plot_range]], [AmpI[plot_range], AmpI_filt[plot_range], AmpI_demodulated[plot_range]],\
                                                      CarrierTones[tt], data_type = 'I', save_path = Plots_dir)
            if debug is not True:
                I_filtered_data=[]
                fI_filt=[]
                fI_demodulated=[]
                AmpI_filt=[]
                AmpI_demodulated=[]
        
        # Q signal
        print('Demodulating Q')
        for tt in range(CarrierTones.shape[0]):
            # Bandpass filtering based on carrier tone frequency
            #if Special_Sweep:
                #Q_filtered_data = butter_lowpass_filter(Q_data, CarrierTones[tt] + BandWidth, fs, Filter_order)#, plots)
            #else:
            Q_filtered_data = butter_bandpass_filter(Q_data, CarrierTones[tt] - BandWidth, CarrierTones[tt] + BandWidth, fs, Filter_order)#, plots)
            
            # FFT on filtered data and store individual results in lists
            fQ_filt, AmpQ_filt, AngQ_filt = FFT(Q_filtered_data[throw:], S_time_throw)
            peaksQ, _ = sig.find_peaks(np.abs(AmpQ_filt), prominence=0.01, distance= 2*BandWidth* S_time_throw)
            print('Peaks found in Q filtered data', peaksQ, 'corresponding to', (fQ_filt[peaksQ] / 1e3), 'kHz')
            if PhaseCorrection is False:
                AngQ_filtered = AngQ_filt[peaksQ[0]]
                Delta_Phase_Q = -(WF_phase_filt[tt]-AngQ_filtered)
                # AngQ_filt = []
                print(f'The phase difference between the WaveForm and Q at the carrier tone {CarrierTones[tt]} is {Delta_Phase_Q} rad')
                # Compute demodulated Q
                # Q_demodulated_data = butter_lowpass_filter(2*phase_shift_signal(WaveForm_filtered[tt][:],fs, shift=Delta_Phase_I/(2*np.pi*fQ_filt[peaks2[tt]])) * Q_filtered_data[throw:], BandWidth, fs, Filter_order) #I assume WF in phase with I
                # Q_demodulated_data = butter_lowpass_filter(2*phase_shift_signal(WaveForm_filtered[tt][:],fs, shift=(np.pi/2)/(2*np.pi*fQ_filt[peaks2[tt]])) * Q_filtered_data[throw:], BandWidth, fs, Filter_order) #I assume WF in phase with I
                Q_demodulated_data = butter_lowpass_filter(2*WaveForm_filtered[tt][:] * Q_filtered_data[throw:], BandWidth, fs, Filter_order) #I assume WF in phase with I
                # Q_demodulated_data = butter_lowpass_filter(-WaveForm_filteredQ[tt][:] * Q_filtered_data[throw:], BandWidth, fs, Filter_order) #I assume WF in phase with I
                print('Phase Correction is not activated, applied a pi/2 phase shift.')
            else:
                AngQ_filtered = AngQ_filt[peaksQ[0]]
                # AngQ_filtered = AngQ_filt[peaks[0]]
                Delta_Phase_Q = -(WF_phase_filt[tt]-AngQ_filtered)#-np.pi/2
                AngQ_filt = []
                print(f'The phase difference between the WaveForm and Q at the carrier tone {CarrierTones[tt]} is {Delta_Phase_Q} rad')
                Delta_Phase_Q = Delta_Phase_I_list[tt]#-np.pi/2
                # Compute demodulated Q
                Q_demodulated_data = butter_lowpass_filter(2*phase_shift_signal(WaveForm_filtered[tt][:],fs,shift=Delta_Phase_Q/( 2*np.pi*fQ_filt[peaksQ[0]] )) * Q_filtered_data[throw:], BandWidth, fs, Filter_order) #I assume WF in phase with I
                print('Phase Correction is activated and applied in demodulation.')
            Q_demodulated_list.append(Q_demodulated_data)
        
            if plots is True: 

                fQ_demodulated, AmpQ_demodulated, AngQ_demodulated = FFT(Q_demodulated_data, len(Q_demodulated_data[throw:])/fs)
                figQ = control_plots_demodulation(timedata[plot_range], [Q_data[plot_range], Q_filtered_data[plot_range], Q_demodulated_data[plot_range]],\
                                                  [fQ[plot_range], fQ_filt[plot_range], fQ_demodulated[plot_range]], [AmpQ[plot_range], AmpQ_filt[plot_range], AmpQ_demodulated[plot_range]],\
                                                      CarrierTones[tt], data_type = 'Q', save_path = Plots_dir)
            if debug is not True:
                Q_filtered_data=[]
                fQ_filt=[]
                fQ_demodulated=[]
                AmpQ_filt=[]
                AmpQ_demodulated=[]
        
        if Noise_debug:
            for tt in range(CarrierTones.shape[0]):
                plot_IQ_scatter(I_demodulated_list[tt][plot_range], I_demodulated_list[tt][plot_range], CarrierTones[tt], data_type='IQ', save_path=Plots_dir)
            
        
        
        if debug is True:
            print('Debug mode ON, oh boy, here we go again')
            # assert CarrierTones.shape[0] == 1, 'Debug only one tone for now'
            # List for plots: Raw, Filtered, Demodulated
            if PhaseCorrection is True:
                Q_plots = [Q_data[throw:], Q_filtered_data[throw:], Q_demodulated_data]
                I_plots = [I_data[throw:], I_filtered_data[throw:], I_demodulated_data]
                WaveForm_plots = [WaveForm_data[throw:], WaveForm_filtered_data[throw:],\
                                  phase_shift_signal(WaveForm_filtered[0][:], fs, shift=Delta_Phase_I_list[0]/( 2*np.pi*fI_filt[peaksI[0]] )),\
                                      phase_shift_signal(WaveForm_filtered[0][:],fs,shift=Delta_Phase_Q/( 2*np.pi*fQ_filt[peaksQ[0]] ))]
            else:
                Q_plots = [Q_data[throw:], Q_filtered_data[throw:], Q_demodulated_data]
                I_plots = [I_data[throw:], I_filtered_data[throw:], I_demodulated_data]
                # WaveForm_plots = [WaveForm_data[throw:], WaveForm_filtered_data[throw:],\
                #                   WaveForm_filtered[0][:],\
                #                       WaveForm_filtered[0][:]]
                WaveForm_plots = [WaveForm_data[throw:], WaveForm_filtered_data[throw:],\
                                  WaveForm_filtered[0][:],\
                                      WaveForm_filtered[0][:]]
            # debug_demodulation_plots(Q_plots, I_plots, WaveForm_plots, timedata[throw:], fs)
            debug_demodulation_plots_separate(Q_plots, I_plots, WaveForm_plots, timedata[throw:], fs)
            
            
            # WaveForm_plots = [WaveForm_data[throw:], WaveForm_filtered_data[throw:],\
            #                       phase_shift_signal(WaveForm_filtered[tt][:],fs,shift=Delta_Phase_Q/( 2*np.pi*fQ_filt[peaks2[tt]] ))]
            # WaveForm_plots = [WaveForm_data[throw:], WaveForm_filtered_data[throw:],\
            #                           phase_shift_signal(WaveForm_filtered[tt][:],fs,shift=Delta_Phase_Q/( 2*np.pi*fQ_filt[peaks2[tt]] ))]
            # for ii in range(len(Q_plots)):
            #     plot_rolling_phase_and_freq(Q_plots[ii], WaveForm_plots[ii], fs, CarrierTones[0], window_size_periods=100)
            # WaveForm_plots = [WaveForm_data[throw:], WaveForm_filtered_data[throw:],\
            #                   phase_shift_signal(WaveForm_filtered[tt][:], fs, shift=Delta_Phase_I_list[0]/( 2*np.pi*fI_filt[peaks2[tt]] ))]
            # for ii in range(len(Q_plots)):
            #     plot_rolling_phase_and_freq(I_plots[ii], WaveForm_plots[ii], fs, CarrierTones[0], window_size_periods=100)
            
            
    return I_demodulated_list, Q_demodulated_list


def normalize_rms_amplitude(time_stream_data):
    """
    Normalize the input time stream data to an RMS amplitude of 1.

    Parameters:
    - time_stream_data (ndarray): Array of waveform data to normalize.

    Returns:
    - normalized_data (ndarray): The time stream data normalized to an RMS amplitude of 1.
    - rms_amplitude (float): The calculated RMS amplitude before normalization.
    """
    # Calculate the RMS amplitude
    rms_amplitude = np.sqrt(np.mean(time_stream_data**2))
    print(f"Calculated RMS amplitude: {rms_amplitude}, normalized to be equal to sqrt(2)")
    scaling_factor = 1 / np.sqrt(2)

    # Normalize the data by dividing by the RMS amplitude
    if rms_amplitude != 0:
        normalized_data = scaling_factor * time_stream_data / rms_amplitude
    else:
        print("Warning: RMS amplitude is zero. Data not normalized.")
        normalized_data = time_stream_data

    return normalized_data#, rms_amplitude


def debug_demodulation_plots(Q_plots, I_plots, WaveForm_plots, timedata, fs):
    """
    Generate a 2x3 grid plot for debugging demodulation by comparing Q, I, and waveform data.
    
    Parameters:
        - Q_plots (list of ndarray): Contains Q data arrays for each stage (Raw, Filtered, Demodulated).
        - I_plots (list of ndarray): Contains I data arrays for each stage (Raw, Filtered, Demodulated).
        - WaveForm_plots (list of ndarray): Contains waveform data arrays, including Raw, Filtered, 
        and phase-shifted versions for I and Q signals. (first I then Q for the 2 last entries)
        - timedata (ndarray): Array of time values for the x-axis.
        - fs (float): Sampling frequency of the signals, used for phase shift calculations (optional if needed for debugging).
    
    Description:
        - Creates a 2x3 grid of subplots. The top row displays Q data, and the bottom row displays I data.
        - Each column represents a different stage: Raw, Filtered, and Demodulated.
        - Within each subplot, Q and I data are plotted for all stages in the background (with reduced opacity).
        - The main Q or I data of each column is plotted on top.
        - Corresponding waveform data for each stage is overlaid with dashed lines in the background.
        
    Note:
        - Legends are included for clarity and are positioned outside the plot area to reduce clutter.
    
    """
    
    # Calculate V_RMS for Q_plots[0] and I_plots[0]
    V_RMS_Q = np.sqrt(np.mean(Q_plots[0]**2))
    V_RMS_I = np.sqrt(np.mean(I_plots[0]**2))
    
    # Calculate V_RMS for WaveForm_plots[3] and WaveForm_plots[2] (assuming they represent Q and I waveforms)
    V_RMS_Waveform_I = np.sqrt(np.mean(WaveForm_plots[3]**2))
    V_RMS_Waveform_Q = np.sqrt(np.mean(WaveForm_plots[2]**2))
    
    # Calculate scaling factors
    scaling_factor_I = V_RMS_I / V_RMS_Waveform_I
    scaling_factor_Q = V_RMS_Q / V_RMS_Waveform_Q
    
    # Apply scaling factors to all elements in WaveForm_plots_Q and WaveForm_plots_I
    WaveForm_plots_Q = [waveform * scaling_factor_Q for waveform in [WaveForm_plots[i] for i in [0, 1, 3]]]
    WaveForm_plots_I = [waveform * scaling_factor_I for waveform in [WaveForm_plots[i] for i in [0, 1, 2]]]
    
    Type_list = ['Raw', 'Filtered', 'Demodulated']
    color_list = ['k', 'r', 'b']  # Set the colors
    fig, axs = plt.subplots(nrows=2, ncols=3, gridspec_kw=dict(hspace=0.5, wspace=0.3), figsize=(16, 9))
    
    for cc in range(3):
        # Plot for I and Q data (row 0 for I, row 1 for Q)
        # I data
        for other in [i for i in range(len(Q_plots)) if i != cc]:
            axs[0][cc].plot(timedata, Q_plots[other], color=color_list[other], markersize=5, alpha=0.2, rasterized=True, label=Type_list[other])
        axs[0][cc].plot(timedata, Q_plots[cc], color=color_list[cc], markersize=5, rasterized=True, label=Type_list[cc])
        axs[0][cc].set_xlabel('Time [s]', fontsize=12)
        axs[0][cc].set_ylabel('Q Data', fontsize=10)
        axs[0][cc].set_title(f'Q Data: {Type_list[cc]}', fontsize=12)
        # Plot other WaveForm_plots_Q with reduced alpha
        for other in [i for i in range(len(WaveForm_plots_Q)) if i != cc]:
            axs[0][cc].plot(timedata, WaveForm_plots_Q[other], color=color_list[other], linestyle='--', alpha=0.1, label=f'Waveform {Type_list[other]}')
        # Add WaveForm data with dashed lines
        axs[0][cc].plot(timedata, WaveForm_plots_Q[cc], color=color_list[cc], linestyle='--', alpha=0.5, label=f'Waveform {Type_list[cc]}')
        
        # Q data
        for other in [i for i in range(len(I_plots)) if i != cc]:
            axs[1][cc].plot(timedata, I_plots[other], color=color_list[other], markersize=5, alpha=0.2, rasterized=True, label=Type_list[other])
        axs[1][cc].plot(timedata, I_plots[cc], color=color_list[cc], markersize=5, rasterized=True, label=Type_list[cc])
        axs[1][cc].set_xlabel('Time [s]', fontsize=12)
        axs[1][cc].set_ylabel('I Data', fontsize=10)
        axs[1][cc].set_title(f'I Data: {Type_list[cc]}', fontsize=12)
        for other in [i for i in range(len(WaveForm_plots_Q)) if i != cc]:
            axs[0][cc].plot(timedata, WaveForm_plots_I[other], color=color_list[other], linestyle='--', alpha=0.1, label=f'Waveform {Type_list[other]}')
        # Add WaveForm data with dashed lines
        axs[1][cc].plot(timedata, WaveForm_plots_I[cc], color=color_list[cc], linestyle='--', alpha=0.5, label=f'Waveform {Type_list[cc]}')

    # Create a combined legend outside the subplots
    handles, labels = axs[0][0].get_legend_handles_labels()  # Collect labels from the first subplot
    # Remove alpha from handles for the legend
    for handle in handles:
        handle.set_alpha(1.0)
    fig.legend(handles, labels, loc='upper center', ncol=len(Type_list), fontsize=12, title='Data Types')
    
    plt.tight_layout()  # Adjust the spacing to prevent overlap
    plt.show()  # Show the plot or save depending on your workflow

def debug_demodulation_plots_separate(Q_plots, I_plots, WaveForm_plots, timedata, fs):
    """
    Generate separate plots for debugging demodulation by comparing Q, I, and waveform data for each stage.

    Parameters:
        - Q_plots (list of ndarray): Contains Q data arrays for each stage (Raw, Filtered, Demodulated).
        - I_plots (list of ndarray): Contains I data arrays for each stage (Raw, Filtered, Demodulated).
        - WaveForm_plots (list of ndarray): Contains waveform data arrays, including Raw, Filtered, 
          and phase-shifted versions for I and Q signals.
        - timedata (ndarray): Array of time values for the x-axis.
        - fs (float): Sampling frequency of the signals.
    """

    # Calculate V_RMS for Q_plots[0] and I_plots[0]
    V_RMS_Q = np.sqrt(np.mean(Q_plots[0]**2))
    V_RMS_I = np.sqrt(np.mean(I_plots[0]**2))

    # Calculate V_RMS for WaveForm_plots[3] and WaveForm_plots[2] (assuming they represent Q and I waveforms)
    V_RMS_Waveform_I = np.sqrt(np.mean(WaveForm_plots[3]**2))
    V_RMS_Waveform_Q = np.sqrt(np.mean(WaveForm_plots[2]**2))

    # Calculate scaling factors
    scaling_factor_I = V_RMS_I / V_RMS_Waveform_I
    scaling_factor_Q = V_RMS_Q / V_RMS_Waveform_Q

    # Apply scaling factors to all elements in WaveForm_plots_Q and WaveForm_plots_I
    WaveForm_plots_Q = [waveform * scaling_factor_Q for waveform in [WaveForm_plots[i] for i in [0, 1, 3]]]
    WaveForm_plots_I = [waveform * scaling_factor_I for waveform in [WaveForm_plots[i] for i in [0, 1, 2]]]

    Type_list = ['Raw', 'Filtered', 'Demodulated']
    Type_list_WF = ['Raw', 'Filtered', 'Shifted']
    color_list = ['k', 'r', 'b']  # Set the colors

    for cc in range(3):
        # Create a new figure for Q data for each stage
        fig_Q, ax_Q = plt.subplots(figsize=(8, 6))
        for other in [i for i in range(len(Q_plots)) if i != cc]:
            ax_Q.plot(timedata, Q_plots[other], color=color_list[other], markersize=5, alpha=0.2, rasterized=True, label=Type_list[other])
        ax_Q.plot(timedata, Q_plots[cc], color=color_list[cc], markersize=5, rasterized=True, label=Type_list[cc])
        ax_Q.set_xlabel('Time [s]', fontsize=12)
        ax_Q.set_ylabel('Q Data', fontsize=10)
        ax_Q.set_title(f'Q Data: {Type_list[cc]}', fontsize=12)
        for other in [i for i in range(len(WaveForm_plots_Q)) if i != cc]:
            ax_Q.plot(timedata, WaveForm_plots_Q[other], color=color_list[other], linestyle='--', alpha=0.1, label=f'Waveform {Type_list_WF[other]}')
        ax_Q.plot(timedata, WaveForm_plots_Q[cc], color=color_list[cc], linestyle='--', alpha=0.5, label=f'Waveform {Type_list_WF[cc]}')
        # Adjust legend placement for Q plots
        ax_Q.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        fig_Q.tight_layout(rect=[0, 0, 0.85, 1])  # Adjusts layout to fit legend outside
        fig_Q.tight_layout()
        fig_Q.show()

        # Create a new figure for I data for each stage
        fig_I, ax_I = plt.subplots(figsize=(8, 6))
        for other in [i for i in range(len(I_plots)) if i != cc]:
            ax_I.plot(timedata, I_plots[other], color=color_list[other], markersize=5, alpha=0.2, rasterized=True, label=Type_list[other])
        ax_I.plot(timedata, I_plots[cc], color=color_list[cc], markersize=5, rasterized=True, label=Type_list[cc])
        ax_I.set_xlabel('Time [s]', fontsize=12)
        ax_I.set_ylabel('I Data', fontsize=10)
        ax_I.set_title(f'I Data: {Type_list[cc]}', fontsize=12)
        for other in [i for i in range(len(WaveForm_plots_I)) if i != cc]:
            ax_I.plot(timedata, WaveForm_plots_I[other], color=color_list[other], linestyle='--', alpha=0.1, label=f'Waveform {Type_list_WF[other]}')
        ax_I.plot(timedata, WaveForm_plots_I[cc], color=color_list[cc], linestyle='--', alpha=0.5, label=f'Waveform {Type_list_WF[cc]}')
        # Adjust legend placement for I plots
        ax_I.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        fig_I.tight_layout(rect=[0, 0, 0.85, 1])  # Adjusts layout to fit legend outside
        fig_I.tight_layout()
        fig_I.show()
        
    # fig_IQ, ax_IQ = plt.subplots(figsize=(8,6))
    # ax_IQ.plot(timedata, I_plots[0], 'r..', markersize=5, alpha = 0.2, rasterized=True)
    # ax_IQ.plot(timedata, Q_plots[0], 'b..', markersize=5, alpha = 0.2, rasterized=True)
    # ax_IQ.plot(timedata, I_plots[0], 'r-', markersize=5, alpha = 0.2, rasterized=True)
    # ax_IQ.plot(timedata, Q_plots[0], 'b..', markersize=5, alpha = 0.2, rasterized=True)

def rolling_phase_difference(x_data, y_data, fs, CarrierTone, window_size_periods=10):
    """
    Calculates the rolling phase difference between two signals in the time domain.
    The calculation is done over a window of data with the specified number of periods of the CarrierTone.

    Parameters:
    - x_data (ndarray): The first signal data (e.g., I or Q signal).
    - y_data (ndarray): The second signal data (e.g., WaveForm signal).
    - fs (float): The sampling frequency in Hz.
    - CarrierTone (float): The frequency of the carrier tone in Hz.
    - window_size_periods (int, optional): The window size in terms of carrier periods. Default is 10 periods.

    Returns:
    - phase_diff (ndarray): The phase difference between the two signals (in radians) for each window.
    - freq_diff (ndarray): The frequency difference between the two signals for each window.
    
    Note:
    - The function assumes that both signals have the same sampling frequency (fs).
    - The phase difference is calculated based on the peak of the carrier frequency in the FFT of each windowed segment.
    """
    
    # Calculate the period of the CarrierTone in seconds
    period = 1 / CarrierTone  # Period in seconds
    
    # Convert the period to samples based on the sampling frequency
    window_size_samples = int(window_size_periods * period * fs)  # Window size in samples
    
    # Initialize lists for storing phase and frequency differences
    phase_diff = []
    freq_diff = []
    
    # Calculate the phase and frequency differences in a rolling window
    for i in range(0, len(x_data) - window_size_samples, window_size_samples):
        # Get the current window of data
        window_x = x_data[i:i + window_size_samples]
        window_y = y_data[i:i + window_size_samples]
        
        # Perform FFT on both signals
        X = np.fft.fft(window_x)
        Y = np.fft.fft(window_y)
        
        # Get the corresponding frequency bins
        freqs = np.fft.fftfreq(len(window_x), 1/fs)
        
        # Find the index of the peak frequency closest to the carrier tone
        peak_freq_index_X = np.argmax(abs(X[1:window_size_samples//2]))+1
        peak_freq_index_Y = np.argmax(abs(Y[1:window_size_samples//2]))+1
        
        # Calculate the phase at the peak frequency for both signals
        phase_x = np.angle(X[peak_freq_index_X])
        phase_y = np.angle(Y[peak_freq_index_Y])
        
        # Get the frequencies corresponding to the peak indices
        freqs_x = freqs[peak_freq_index_X]
        freqs_y = freqs[peak_freq_index_Y]
        
        # Calculate the phase difference (in radians) between the two signals
        phase_diff_value = phase_y - phase_x
        phase_diff.append(phase_diff_value)
        
        # Calculate the frequency difference between the two signals
        freq_diff_value = freqs_y - freqs_x
        freq_diff.append(freq_diff_value)
    
    # Convert the lists to numpy arrays
    phase_diff = np.array(phase_diff)
    freq_diff = np.array(freq_diff)
    
    return phase_diff, freq_diff

def plot_rolling_phase_and_freq(IQ_data, WaveForm_data, fs, CarrierTone, window_size_periods=10):
    """
    Plots the rolling phase difference and frequency difference over time between two signals.
    
    Parameters:
    - IQ_data (ndarray): The first signal data (e.g., I or Q signal).
    - WaveForm_data (ndarray): The second signal data (e.g., WaveForm signal).
    - fs (float): The sampling frequency in Hz.
    - CarrierTone (float): The frequency of the carrier tone in Hz.
    - window_size_periods (int, optional): The window size in terms of carrier periods. Default is 10 periods.
    
    This function calls `rolling_phase_difference` to compute the rolling phase and frequency differences,
    and then plots them.
    """
    # Call the rolling_phase_difference function
    phase_diff, freq_diff = rolling_phase_difference(IQ_data, WaveForm_data, fs, CarrierTone, window_size_periods)
    
    # Create the time vector for the phase and frequency differences
    time_vector = np.arange(0, len(phase_diff)) * (window_size_periods / CarrierTone)  # Time for each window
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot phase difference in radians
    ax1.set_ylabel('Phase Difference (rad)', fontsize=12)
    ax1.plot(time_vector, phase_diff, color='tab:blue', label='Phase Difference')
    ax1.set_title('Rolling Phase Difference Over Time', fontsize=14)
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    # Plot frequency difference in Hz
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Frequency Difference (Hz)', fontsize=12)
    ax2.plot(time_vector, freq_diff, color='tab:red', label='Frequency Difference')
    ax2.set_title('Rolling Frequency Difference Over Time', fontsize=14)
    ax2.grid(True)
    ax2.legend(loc='upper right')
    
    # Adjust layout to prevent overlap
    fig.tight_layout()
    
    # Show the plot
    plt.show()

def plot_IQ_scatter(
    I_data, 
    Q_data, 
    Carrier_Frequency, 
    data_type='IQ', 
    save_path=None
):
    """
    Generate and optionally save or display an IQ scatter plot.

    Parameters:
    - I_data (ndarray): Array of I data.
    - Q_data (ndarray): Array of Q data.
    - Carrier_Frequency (float): Carrier frequency in Hz, used for labeling the plot.
    - data_type (str): Data type label for plot title, default is 'IQ'.
    - save_path (str or None): Directory path to save the plot as a PDF. If None, displays the plot instead of saving.

    Returns:
    - fig (matplotlib.figure.Figure): The created figure, if displaying instead of saving.
    
    Notes:
    - The function saves the file with the name 'IQ_{Carrier_Frequency_kHz}kHz_ScatterPlot.pdf',
      where {Carrier_Frequency_kHz} is the carrier frequency rounded to the nearest kHz.
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))  # Square aspect ratio for IQ scatter

    # Scatter plot for I vs Q
    ax.plot(I_data, Q_data, 'o', markersize=2, color='blue', alpha=0.6, rasterized=True)
    ax.set_xlabel('I (In-phase)', fontsize=12)
    ax.set_ylabel('Q (Quadrature)', fontsize=12)
    ax.set_title(f'{data_type} IQ Scatter Plot @ {Carrier_Frequency / 1e3:.0f} kHz', fontsize=14)
    # ax.axis('equal')
    ax.grid(True)
    
    # # Enforcing orthonormal scales
    # max_range = max(np.max((I_data)), np.max((Q_data)))
    # min_range = max(np.min((I_data)), np.min((Q_data)))
    # avg_range = (max_range - min_range)/2
    # ax.set_xlim(avg_range-max_range, avg_range+max_range)
    # ax.set_ylim(avg_range-max_range, avg_range+max_range)
    ax.set_aspect('equal', 'box')
    
    # Saving or displaying the plot
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        carrier_kHz = round(Carrier_Frequency / 1e3)
        save_filename = f'{save_path}IQ_{carrier_kHz}kHz_ScatterPlot.pdf'
        
        with PdfPages(save_filename) as pdf:
            pdf.savefig(fig)
            print(f'Saved the IQ scatter plot to {save_filename}')
        plt.close(fig)  # Close figure after saving
    else:
        plt.show()  # Display the plot if no save path is provided

    return fig if save_path is None else None

def control_plots_demodulation(
    timedata, 
    TimeStream_data, 
    Frequencies_data, 
    Amplitudes_data,
    Carrier_Frequency,
    data_type='Unknown Type', 
    save_path=None
):
    """
    Generate and optionally save or display control plots for demodulation.

    Parameters:
    - timedata (ndarray): Array of time data.
    - TimeStream_data (list of ndarrays): List of time-domain data arrays for different steps (e.g., raw, filtered, demodulated).
    - Frequencies_data (list of ndarrays): List of frequency data arrays for different steps (matching TimeStream_data).
    - Amplitudes_data (list of ndarrays): List of amplitude data arrays corresponding to Frequencies_data.
    - Carrier_Frequency (float): Carrier frequency in Hz, used as a reference marker in frequency plots.
    - data_type (str): Data type label (e.g., 'I', 'Q', 'WaveForm') for plot labeling.
    - save_path (str or None): Directory path to save the plots as a PDF. If None, displays the plots instead of saving.

    Returns:
    - fig (matplotlib.figure.Figure): The created figure, if displaying instead of saving.
    
    Notes:
    - The function saves the file with the name 'Demodulation_{Carrier_Frequency_kHz}kHz_ControlPlots.pdf',
      where {Carrier_Frequency_kHz} is the carrier frequency rounded to the nearest kHz.
    """
    
    assert len(TimeStream_data) == len(Frequencies_data) == len(Amplitudes_data), \
        "The number of steps for plots don't match each other."

    Type_list = ['Raw', 'Filtered', 'Demodulated']
    color_list = ['k', 'r', 'b']
    
    fig, axs = plt.subplots(nrows=2, ncols=len(TimeStream_data), gridspec_kw=dict(hspace=0.5, wspace=0.3), figsize=(16, 9))
    
    for cc in range(len(TimeStream_data)):
        
        # Plot TimeStream_data
        for other in [i for i in range(len(TimeStream_data)) if i != cc]:
            axs[0][cc].plot(timedata, TimeStream_data[other], color=color_list[other], markersize=5, alpha=0.2, rasterized=True, label=Type_list[other])
        axs[0][cc].plot(timedata, TimeStream_data[cc], color=color_list[cc], markersize=5, rasterized=True, label=Type_list[cc])
        axs[0][cc].set_xlabel('Time [s]', fontsize=12)
        axs[0][cc].set_ylabel(data_type + ' Data', fontsize=10)
        # axs[0][cc].legend()
        
        # Plot Frequency-Amplitude data
        for other in [i for i in range(len(TimeStream_data)) if i != cc]:
            axs[1][cc].plot(Frequencies_data[other], Amplitudes_data[other], color=color_list[other], markersize=5, alpha=0.2, rasterized=True, label=Type_list[other])
        axs[1][cc].plot(Frequencies_data[cc], Amplitudes_data[cc], color=color_list[cc], markersize=5, alpha=1, rasterized=True, label=Type_list[cc])
        axs[1][cc].axvline(x=Carrier_Frequency, color='green', linestyle='--', linewidth=1)
        axs[1][cc].set_xlabel('Frequency [Hz]', fontsize=12)
        axs[1][cc].set_ylabel('Amplitude (' + data_type + ' FFT)', fontsize=10)
        # axs[1][cc].legend()
    
    # Create a combined legend outside the subplots
    handles, labels = axs[0][0].get_legend_handles_labels()  # Collect labels from the first subplot
    # Remove alpha from handles for the legend
    for handle in handles:
        handle.set_alpha(1.0)
    fig.legend(handles, labels, loc='upper center', ncol=len(Type_list), fontsize=12, title='Data Types')
    
    # Adjust layout to make room for the legend
    plt.subplots_adjust(top=0.85)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        # Round the carrier frequency to the nearest kHz and save all figures in a single PDF
        carrier_kHz = round(Carrier_Frequency / 1e3)
        save_filename = f'{save_path}Demodulation_{data_type}_{carrier_kHz}kHz_ControlPlots.pdf'
        
        with PdfPages(save_filename) as pdf:
            pdf.savefig(fig)
            print(f'Saved the control plots for {data_type} data to {save_filename}')
        plt.close(fig)  # Close figure after saving
    else:
        plt.show()  # Display the plot if no save path is provided
        
    return fig if save_path is None else None

def phase_shift_signal(signal, fs, shift=np.pi/2):
    """
    Applies a phase shift to a signal in the frequency domain.

    Parameters:
    - signal (ndarray): 1D array of the time-domain signal to be phase-shifted.
    - fs (float): Sampling frequency of the signal in Hz.
    - shift (float): Phase shift in radians (default is np.pi/2).

    Returns:
    - shifted_signal (ndarray): The time-domain signal after applying the phase shift.
    """
    # Perform FFT on the original signal
    N = len(signal)
    freq_domain = np.fft.fft(signal)
    
    # Generate the frequency bins
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Apply the phase shift in the frequency domain
    freq_domain_shifted = freq_domain * np.exp(1j * 2*np.pi*freqs * shift)
    
    # Transform back to the time domain with iFFT
    shifted_signal = np.fft.ifft(freq_domain_shifted).real
    
    return shifted_signal


def process_twotonedata_onetonemode(dir_path):
    
    root = tk.Tk()
    file_path = filedialog.askopenfilename(title='Choose .h5 file with Sdff data', initialdir=dir_path)
    filename = file_path.split('/')[:-3]
    root.destroy()
    main_dict = {}
    meas_dict={}
    resfit_dict = {'Qi':np.zeros([1]), 'Qc':np.zeros([1]), 'Qr':np.zeros([1]), 'fres':np.zeros([1])}
    
    with h5py.File(file_path, 'r') as f:
        key_BBtemp = list(f.keys())[0]
        key_temp = list(f[key_BBtemp].keys())[0]
        key_pwr = list(f[key_BBtemp][key_temp].keys())[0]
        for key in f[key_BBtemp][key_temp][key_pwr]['res_scan']['data'].attrs.keys():
            main_dict[key] = f[key_BBtemp][key_temp][key_pwr]['res_scan']['data'].attrs[key]
        for key in f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['meas_dict'].attrs.keys():
            meas_dict[key] = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['meas_dict'].attrs[key]
        for key in resfit_dict.keys():
            resfit_dict[key] = f[key_BBtemp][key_temp][key_pwr]['res_scan']['fit'].attrs[key]
        angle = f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['angle'][()]
        dIQ_vs_df = f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['dIQ_vs_df'][()]
        L = len(f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise'].keys())
        
        N_noise = int((L-1)/2)
        fs = meas_dict['DAQ_freq']
        Npts = meas_dict['DAQ_Npts']
        S_time = Npts/fs
        # timedata = np.arange(0, S_time, 1/fs)
        nfactor = 50
        nperseg = int(Npts/nfactor)
        LP_cutoff = 200000
        
        CarrierTones = np.array([100e3])
        
        # for i in range(N_noise):
        i=0
        # # f_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['freq_data'][()]
        # # I_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['I_onres%d'%i][()]
        # # Q_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['Q_onres%d'%i][()]
        # # m_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['WaveFormG'][()]
        # # I_offdata = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['I_offres%d'%i][()]
        # # Q_offdata = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['Q_offres%d'%i][()]
        # f_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['f'][()]
        # PSD_onres_freq_avg = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['PSD_onres_freq_avg'][()]
        # PSD_onres_diss_avg = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['PSD_onres_diss_avg'][()]
        # # PSD_offres_freq_avg = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['PSD_offres_freq_avg'][()]
        # # PSD_offres_diss_avg = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['PSD_offres_diss_avg'][()]
        
        Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
        
        # f1 = np.array([70e3])
        Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
        filename = Sdf_dir + 'OneTones_Sdff'
        
        tone1_I, tone1_Q = numerical_demodulation(
            f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['I_onres%d'%i][()],
            f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['Q_onres%d'%i][()], fs, CarrierTones, 50e3, Filter_order = 5,
            WaveForm_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['WaveFormG'][()],
            PhaseCorrection = False, plots = False, Plots_dir = Sdf_dir, Noise_debug = True)
        # tone1_I, tone1_Q = numerical_demodulation(
        #     I_data ,
        #     Q_data, fs, CarrierTones, 30e3,
        #     WaveForm_data = m_data,
        #     PhaseCorrection = True, plots = True)#, Plots_dir = Sdf_dir)
        
        tone1_I = tone1_I[0]
        tone1_Q = tone1_Q[0]
        
        # f_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['freq_data'][()]
        # I_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['I_onres%d'%i][()]
        # Q_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['Q_onres%d'%i][()]
        # m_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['WaveFormG'][()]
        # I_offdata = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['I_offres%d'%i][()]
        # Q_offdata = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['Q_offres%d'%i][()]
        f_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['f'][()]
        PSD_onres_freq_avg = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['PSD_onres_freq_avg'][()]
        PSD_onres_diss_avg = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['PSD_onres_diss_avg'][()]
        # PSD_offres_freq_avg = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['PSD_offres_freq_avg'][()]
        # PSD_offres_diss_avg = f[key_BBtemp][key_temp][key_pwr]['res_scan']['Sdff']['PSD_offres_diss_avg'][()]
    
        # fI, AmpI, AngI = FFT(I_data,S_time)
        # fQ, AmpQ, AngQ = FFT(Q_data,S_time)
        # # fm, Ampm, Angm = FFT(m_data,S_time)
        # peaks, _ = sig.find_peaks(np.abs(AmpI+1j*AmpQ), prominence=0.01, distance=200e3*S_time)
        # print(peaks)
        
        
        # data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(I_data, Q_data, angle, dIQ_vs_df, resfit_dict['fres'])
        # f_data, Sdff_onres_freq = sig.welch(data_noise_onres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
        # _, Sdff_onres_diss = sig.welch(data_noise_onres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
        
        data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(tone1_I, tone1_Q, angle, dIQ_vs_df, resfit_dict['fres'])
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
        # axs[0].relim()
        axs[0].set_ylim([1e-21, 1e-13])
        axs[0].autoscale_view()  
        figtitle = '%s:  %dmK  fitres %sMHz'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), meas_dict['freq_center_str'])
        fig.text(0.5, 0.95, figtitle, fontsize=12, horizontalalignment='center', verticalalignment='top')
        
        timestream_factor = 0.1
        decimation_factor = 1
        
        
        # print(np.mean(Sdff_onres_freq[int(200*S_time/nfactor):int(1000*S_time/nfactor)]))
        # print(np.mean(Sdff_onres_diss[int(30000*S_time/nfactor):int(80000*S_time/nfactor)]))

        # # data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(I_offdata, Q_offdata, angle, dIQ_vs_df, resfit_dict['fres']+main_dict['df_onoff_res'])
        # # f_data, Sdff_onres_freq = sig.welch(data_noise_onres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
        # # _, Sdff_onres_diss = sig.welch(data_noise_onres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
        # data_noise_offres_freq, data_noise_offres_diss = rotate_noise_IQ2FD(tone2_I, tone2_Q, angle, dIQ_vs_df, resfit_dict['fres']+main_dict['df_onoff_res'])
        # f_data, Sdff_offres_freq = sig.welch(data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
        # _, Sdff_offres_diss = sig.welch(data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
        # # data_noise_offres_freq, data_noise_offres_diss = rotate_noise_IQ2FD(tone1_Ioff, tone1_Qoff, angle, dIQ_vs_df, resfit_dict['fres']+main_dict['df_onoff_res'])
        # # f_data, Sdff_offres_freq = sig.welch(data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
        # # _, Sdff_offres_diss = sig.welch(data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
        
        
        # axs[1].loglog(f_data, Sdff_offres_freq, linewidth=2, color='C0', label = 'offres: freq')
        # axs[1].loglog(f_data, Sdff_offres_diss, linewidth=2, color='C1', label = 'offres: diss')
        # axs[1].tick_params(axis='y', labelsize=14)
        # axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
        # axs[1].tick_params(axis='x', labelsize=14)
        # axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
        # axs[1].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
        # axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
        # axs[1].relim()
        # # axs[0].set_ylim(ylims)
        # axs[1].autoscale_view() 
            
        # Low_freq=10000
        # result1=np.corrcoef(Sdff_onres_freq[:int(Low_freq*S_time)], Sdff_offres_freq[:int(Low_freq*S_time)])
        # result2=np.corrcoef(Sdff_onres_diss[:int(Low_freq*S_time)], Sdff_offres_diss[:int(Low_freq*S_time)])
        
        
        # fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
        # axs[0].loglog(f_data, np.abs(PSD_onres_freq_avg), linewidth=2, color='C0', linestyle=':', label = 'calib onres: freq')
        # axs[0].loglog(f_data, np.abs(PSD_onres_diss_avg), linewidth=2, color='C1', linestyle=':', label = 'calib onres: diss')
        # # axs[0].loglog([meas_dict['f_cutoff'], meas_dict['f_cutoff']], [1e-20, 1e-16], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
        
        # axs[0].tick_params(axis='y', labelsize=14)
        # axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
        # axs[0].tick_params(axis='x', labelsize=14)
        # axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
        # axs[0].set_ylabel(r'Sdf/f onres [$Hz^{-1}$]', fontsize = 16)
        # axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
        # axs[0].autoscale_view() 
        
        # axs[1].loglog(f_data, np.abs(PSD_offres_freq_avg), linewidth=2, color='C0', linestyle=':', label = 'calib offres: freq')
        # axs[1].loglog(f_data, np.abs(PSD_offres_diss_avg), linewidth=2, color='C1', linestyle=':', label = 'calib offres: diss')
        # # axs[0].loglog([meas_dict['f_cutoff'], meas_dict['f_cutoff']], [1e-20, 1e-16], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
        
        # axs[1].tick_params(axis='y', labelsize=14)
        # axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
        # axs[1].tick_params(axis='x', labelsize=14)
        # axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
        # axs[1].set_ylabel(r'Sdf/f offres [$Hz^{-1}$]', fontsize = 16)
        # axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
        # axs[1].autoscale_view() 
        
        # fs=2e6
        # n=1
        # for i in range(n):
            
        #     xI, resultI = FFT(fs, I_data[int(i*(fs/n)):int((i+1)*(fs/n))])
        #     xQ, resultQ = FFT(fs, Q_data[int(i*(fs/n)):int((i+1)*(fs/n))])
        #     peaks, amp = sig.find_peaks(resultI, prominence=0.1, distance=200e3)
        #     print(i,peaks, amp)
        #     peaks, amp = sig.find_peaks(resultQ, prominence=0.1, distance=200e3)
        #     print(i,peaks, amp)
        
        #     fig, axs = plt.subplots(nrows=4, ncols=1, gridspec_kw=dict(hspace=0.5, wspace=0.2), figsize=(16, 9))
            
        #     axs[0].plot(I_data[int(i*(fs/n)):int((i+1)*(fs/n))], '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True,label='two-tone setup')
        #     axs[0].set_xlabel('time [us]', fontsize = 12)
        #     axs[0].set_ylabel('I [V]', fontsize = 12)
        #     axs[1].plot(Q_data[int(i*(fs/n)):int((i+1)*(fs/n))], '.', color='C%d'%1, markersize=3, alpha=1, rasterized=True)
        #     axs[1].set_xlabel('time [us]', fontsize = 12)
        #     axs[1].set_ylabel('Q [V]', fontsize = 12)
        #     axs[2].plot(xI, resultI, '.', color='C%d'%0, markersize=3, alpha=1, rasterized=True)
        #     axs[2].set_xlabel('frequency [Hz]', fontsize = 12)
        #     axs[2].set_ylabel('amplitude (I FFT)', fontsize = 12)
        #     axs[3].plot(xQ, resultQ, '.', color='C%d'%1, markersize=3, alpha=1, rasterized=True)
        #     axs[3].set_xlabel('frequency [Hz]', fontsize = 12)
        #     axs[3].set_ylabel('amplitude (Q FFT)', fontsize = 12)
        # fig.legend()

def process_twotonedata_twotonemode(dir_path, h5_filepath = None, I_demodulated = True, Q_demodulated = True):
    root = tk.Tk()
    if h5_filepath is not None:
        file_path = h5_filepath
    else:
        file_path = filedialog.askopenfilename(title='Choose .h5 file with Sdff data', initialdir=dir_path)
    filename = file_path.split('/')[:-3]
    root.destroy()
    main_dict = {}
    meas_dict={}
    Carriers_dict = {'Modulation_Tones':np.zeros([2])}
    Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
    resfit_dict = {'Qi':np.zeros([1]), 'Qc':np.zeros([1]), 'Qr':np.zeros([1]), 'fres':np.zeros([1])}
    resfit_dict_tone2 = {'Qi':np.zeros([1]), 'Qc':np.zeros([1]), 'Qr':np.zeros([1]), 'fres':np.zeros([1])}
    h5_output_filepath = os.path.join(Sdf_dir, f'IQ_demodulated.h5')
    with h5py.File(file_path, 'r') as f:
        print(list(f.keys()))
        key_BBtemp = list(f.keys())[0]
        key_temp = list(f[key_BBtemp].keys())[0]
        key_pwr = list(f[key_BBtemp][key_temp].keys())[0]
        print(key_pwr)
        for key in f[key_BBtemp][key_temp][key_pwr]['res_scan']['data'].attrs.keys():
            main_dict[key] = f[key_BBtemp][key_temp][key_pwr]['res_scan']['data'].attrs[key]
        for key in f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['meas_dict'].attrs.keys():
            meas_dict[key] = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['meas_dict'].attrs[key]
        for key in resfit_dict.keys():
            resfit_dict[key] = f[key_BBtemp][key_temp][key_pwr]['res_scan']['fit'].attrs[key]
        for key in Carriers_dict:
            CarrierTones = f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib'].attrs[key]
        angle = f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['angle'][()]#-.896
        angle = -angle+0.2
        #angle = -1.64
        dIQ_vs_df = f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['dIQ_vs_df'][()]#3.327*10^-5
        L = len(f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise'].keys())

        N_noise = int((L-1)/3)
        print(N_noise)
        fs = meas_dict['DAQ_freq']/10
        Npts = meas_dict['DAQ_Npts']#*10#125000000
        S_time = Npts/fs
        timedata = np.arange(0, S_time, 1/fs)
        nfactor = 100
        nperseg = int(Npts/nfactor)
        LP_cutoff = 30000
        Filter_order=6
        #i=0

        print(f"The fitted frequency is {resfit_dict['fres']*1e-6} MHz")
        print(f"The Calibration angle is {angle} rad")

        Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
        filename_sdff = Sdf_dir + 'TwoTones_Sdff'              
        tone1_I = np.array([])
        tone1_Q = np.array([])
        tone2_I = np.array([])
        tone2_Q = np.array([])
        if I_demodulated and Q_demodulated:
            with h5py.File(h5_output_filepath, 'r') as hf:
                for i in range(10):
                    assert os.path.exists(h5_output_filepath), "h5 file does not exist"
                    print(f"Loading pre-existing demodulated data from: {h5_output_filepath}")
                    try:
                        with h5py.File(h5_output_filepath, 'r') as hf:
                            tone1_I = np.append(tone1_I, hf['tone1_I_demodulated%d'%i][100:])
                            tone2_I = np.append(tone2_I, hf['tone2_I_demodulated%d'%i][100:])
                            tone1_Q = np.append(tone1_Q, hf['tone1_Q_demodulated%d'%i][100:])
                            tone2_Q = np.append(tone2_Q, hf['tone2_Q_demodulated%d'%i][100:])
                    except Exception as e:
                        print(f"Error loading H5 file: {e}.")

        else:
            with h5py.File(h5_output_filepath, 'w') as hf:
                for i in range(N_noise):
                    print("Performing numerical demodulation.")
                    I, Q = numerical_demodulation(
                        f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['I_onres%d'%i][()],
                        f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['Q_onres%d'%i][()], fs, CarrierTones, LP_cutoff,
                        Filter_order = Filter_order, WaveForm_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['WaveFormG%d'%i][()],
                        PhaseCorrection = False, plots = False, Plots_dir = Sdf_dir, Noise_debug = False)
        
                
                    tone1_I = np.append(tone1_I, I[0])
                    tone1_Q = np.append(tone1_Q, Q[0])
                    tone2_I = np.append(tone2_I, I[1])
                    tone2_Q = np.append(tone2_Q, I[1])
                    hf.create_dataset('tone1_I_demodulated%d'%i, data=I[0][::N_noise])
                    hf.create_dataset('tone1_Q_demodulated%d'%i, data=Q[0][::N_noise])
                    hf.create_dataset('tone2_I_demodulated%d'%i, data=I[1][::N_noise])
                    hf.create_dataset('tone2_Q_demodulated%d'%i, data=Q[1][::N_noise])
                    print(f"Newly demodulated data saved to: {h5_output_filepath}")  
       

        theta1 = np.angle(tone1_I + 1j*tone1_Q).mean()
        theta2 = np.angle(tone2_I + 1j*tone2_Q).mean()
        
        print(f"The measured angle of demodulated I and Q is {theta1} rad at {CarrierTones[0]/1e3} kHz and {theta2} rad at {CarrierTones[1]/1e3} kHz, the calib angle is {angle} rad")
        
        No_scale = False

        if No_scale:
            data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD_nonescale(tone1_I, tone1_Q, angle)
        else:
            data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(tone1_I, tone1_Q, angle, dIQ_vs_df, resfit_dict['fres'])
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
        figtitle = '%s:  %dmK  fitres %sMHz'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), meas_dict['freq_center_str'])
        fig.text(0.5, 0.95, figtitle, fontsize=12, horizontalalignment='center', verticalalignment='top')
        if No_scale:
            data_noise_offres_freq, data_noise_offres_diss = rotate_noise_IQ2FD_nonescale(tone2_I, tone2_Q, angle)
        else:
            data_noise_offres_freq, data_noise_offres_diss = rotate_noise_IQ2FD(tone2_I, tone2_Q, angle, dIQ_vs_df, resfit_dict['fres']+main_dict['df_onoff_res'])
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
            


def plot_correlation(x_data, y_data, title, xlabel, ylabel, SavePlot=None, Throw=False):
    # Handle "Throw" option: Remove 10% of the beginning and end of the data
    if Throw:
        num_points = len(x_data)
        start_idx = int(0.1 * num_points)  # 10% from start
        end_idx = int(0.9 * num_points)  # 90% of the total (removes last 10%)
        
        x_data = x_data[start_idx:end_idx]
        y_data = y_data[start_idx:end_idx]

    # Compute Pearson correlation coefficient
    r, _ = pearsonr(x_data, y_data)

    # Perform Linear Fit (Best-Fit Line)
    m, b = np.polyfit(x_data, y_data, 1)  # Linear regression (y = m*x + b)

    # Generate fitted line
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = m * x_fit + b

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_data, y_data, 'b-', label="Data Points", alpha=0.7, linewidth=2)  # Blue dots
    ax.plot(x_fit, y_fit, 'r-', label=f"Fit: y = {m:.2f}x + {b:.2f} (Pearson r = {r:.2f})", linewidth=2)

    # Formatting
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True)
    ax.set_aspect('equal')  # Equal scaling for x and y axes
    ax.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=14)  # Moves legend outside
    
    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Find the largest range
    xy_min = min(x_min, y_min)
    xy_max = max(x_max, y_max)
    
    # Apply same limits to both axes
    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)

    # Save the plot if SavePlot is provided
    if SavePlot is not None:
        print('Saving the Plot:', SavePlot)
        fig.savefig(SavePlot + '.png', dpi=100, bbox_inches='tight')

    # Show the plot
    plt.show()

def extract_noise(D, S, fs, f_cutoff, order, iteration=1, domain="Time", SpectralLines_Filtering=None):
    """
    Extracts intrinsic noise 'n' from the observed signal 'D' using either:
    - Fourier-domain noise cleaning (alpha estimated in frequency domain)
    - Time-domain noise cleaning (alpha estimated using least squares)
    
    Parameters:
    -----------
    D : np.ndarray (complex)
        The observed signal containing both the reference structure and noise.
    S : np.ndarray (complex)
        The reference signal representing the shared component in D.
    fs : float
        Sampling frequency.
    f_cutoff : float
        Frequency cutoff for filtering.
    order : float
        Filter order for filtering.
    iteration : int
        Number of times to iterate noise removal.
    domain : str
        "Fourier" (default) or "Time" - determines how alpha is calculated.

    Returns:
    --------
    n_est : np.ndarray (complex)
        The estimated intrinsic noise (D minus the shared structure).
    """

    # Initial noise estimate is just D
    n_est = D.copy()
    S_rms = np.sqrt(np.mean(S)**2)
    D_rms = np.sqrt(np.mean(D)**2)
    
    S = S/D_rms
    D = D/D_rms
    

    # Apply low pass filter
    D_filtered = butter_lowpass_filter(D, f_cutoff, fs, order= order)
    
    S_filtered = butter_lowpass_filter(S, f_cutoff, fs, order= order)


    for _ in range(iteration):
        if domain == "Fourier":
            # Compute Fourier transforms
            S_f = np.fft.fft(S)
            D_f = np.fft.fft(n_est)  # Work on the residual noise after previous iterations
            freqs = np.fft.fftfreq(len(S), 1/fs)
            
            Forbiden_freqs = []
            if SpectralLines_Filtering is not None:
                for ii in range(SpectralLines_Filtering.shape[0]):
                    if freqs[ii]>SpectralLines_Filtering[ii,0] and freqs[ii]<SpectralLines_Filtering[ii,1]:
                        Forbiden_freqs.append(freqs[ii])

            # Estimate alpha(f) in the Fourier domain
            # Norm = np.sum(np.conj(S_f) * S_f)
            alpha_est = (np.conj(S_f) * D_f + np.conj(D_f) * S_f)
            alpha_f = (np.conj(S_f) * D_f + np.conj(D_f) * S_f) / (2*np.conj(S_f) * S_f)

            # Remove NaNs or infinities due to zero division
            alpha_f[np.isnan(alpha_f)] = 0
            alpha_f[np.isinf(alpha_f)] = 0
            
            alpha_est_sum = 0
            Norm = 0
            D_shared_f = S.copy()
            for ii,ff in enumerate(freqs):
                if ff<f_cutoff and ff not in Forbiden_freqs:
                    alpha_est_sum += alpha_est[ii]
                    Norm += (np.conj(S_f[ii]) * S_f[ii])
                    # Compute shared component in the frequency domain
                    # D_shared_f[ii] = alpha_f[ii] * S_f[ii]
            D_shared_f = alpha_est_sum * S_f / Norm

            # Convert back to time domain
            D_shared = np.fft.ifft(D_shared_f)
            #n_est = D - D_shared  # Noise extracted
            n_est = D - alpha_est * S  # Noise extracted

        elif domain == "Time":
            # Estimate alpha using time-domain least squares
            alpha_est = (np.dot(np.conj(S_filtered), D_filtered)) / (np.dot(np.conj(S_filtered), S_filtered))


            #We then normalize S to the same magnitude as D
            


            D_shared = alpha_est * S_filtered

            # Remove shared component to get noise estimate
            n_est = (D - D_shared)*D_rms
            print(alpha_est)
            return n_est, alpha_est

        else:
            raise ValueError("Invalid domain. Choose 'Fourier' or 'Time'.")

    return n_est, alpha_f, alpha_est_sum, freqs  # Return the cleaned noise signal

def process_twotonedata_TwoResMode(dir_path, h5_filepath = None):
    
    root = tk.Tk()
    if h5_filepath is not None:
        file_path = h5_filepath
    else:
        file_path = filedialog.askopenfilename(title='Choose .h5 file with Sdff data', initialdir=dir_path)
    filename = file_path.split('/')[:-3]
    root.destroy()
    main_dict = {}
    meas_dict={}
    Carriers_dict = {'Modulation_Tones':np.zeros([2])}
    resfit_dict = {'Qi':np.zeros([1]), 'Qc':np.zeros([1]), 'Qr':np.zeros([1]), 'fres':np.zeros([1])}
    resfit_dict_tone2 = {'Qi':np.zeros([1]), 'Qc':np.zeros([1]), 'Qr':np.zeros([1]), 'fres':np.zeros([1])}
    
    with h5py.File(file_path, 'r') as f:
        key_BBtemp = list(f.keys())[0]
        key_temp = list(f[key_BBtemp].keys())[0]
        key_pwr = list(f[key_BBtemp][key_temp].keys())[0]
        for key in f[key_BBtemp][key_temp][key_pwr]['res_scan']['data'].attrs.keys():
            main_dict[key] = f[key_BBtemp][key_temp][key_pwr]['res_scan']['data'].attrs[key]
        for key in f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['meas_dict'].attrs.keys():
            meas_dict[key] = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['meas_dict'].attrs[key]
        for key in resfit_dict.keys():
            resfit_dict[key] = f[key_BBtemp][key_temp][key_pwr]['res_scan']['fit'].attrs[key]
        angle = f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['angle'][()]
        angle = -angle
        dIQ_vs_df = f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['dIQ_vs_df'][()]
        for key in Carriers_dict:
            CarrierTones = f[key_BBtemp][key_temp][key_pwr]['res_scan']['calib'].attrs[key]
        L = len(f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise'].keys())
        
        dir_path = "/".join(file_path.split("/")[:-1])
        new_filename = "off_res" + file_path.split("/")[-1]
        file_path_tone2 = f"{dir_path}/{new_filename}"

        with h5py.File(file_path_tone2, 'r') as f2:
            for key in resfit_dict.keys():
                resfit_dict_tone2[key] = f2[key_BBtemp][key_temp][key_pwr]['res_scan']['fit'].attrs[key]
            angle_tone2 = f2[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['angle'][()]
            angle_tone2 = -angle_tone2
            dIQ_vs_df_tone2 = f2[key_BBtemp][key_temp][key_pwr]['res_scan']['calib']['dIQ_vs_df'][()]
        
        N_noise = int((L-1)/2)
        fs = meas_dict['DAQ_freq']
        Npts = meas_dict['DAQ_Npts']
        S_time = Npts/fs
        timedata = np.arange(0, S_time, 1/fs)
        nfactor = 15
        nperseg = int(Npts/nfactor)
        LP_cutoff = 40000
        # Deltaf = (3.1303379948263425e8-3.127347161200649e8)/2
        print(f"The carriers tones are: {CarrierTones}")
        # CarrierTones = np.array([0.25e6-Deltaf,0.25e6+Deltaf])
        # CarrierTones = np.array([70*1e3,4*70*1e3]) #kHz
        # CarrierTones = np.array([(600-315)*1e3,(600+315)*1e3])
        
        print(f"The fitted frequency are {resfit_dict['fres']*1e-6} MHz and {resfit_dict_tone2['fres']*1e-6} MHz")
        print(f"The Calibration angle are {angle} rad and {angle_tone2}")
        # angle = np.pi/4
        Filter_order=5
        # for i in range(N_noise):
        i=0

        Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
        filename = Sdf_dir + 'TwoTones_Sdff'
        
        # f1 = np.array([70e3])
        
        I, Q = numerical_demodulation(
            f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['I_onres%d'%i][()],
            -f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['Q_onres%d'%i][()], fs, CarrierTones, LP_cutoff,
            Filter_order = Filter_order, WaveForm_data = f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['WaveFormG'][()],
            PhaseCorrection = False, plots = True, Plots_dir = Sdf_dir, Noise_debug = True)
        
        tone1_I = I[0]
        tone1_Q = Q[0]
        tone2_I = I[1]
        tone2_Q = Q[1]
        
        I=[]
        Q=[]
        
        theta1 = np.angle(tone1_I + 1j*tone1_Q).mean()
        theta2 = np.angle(tone2_I + 1j*tone2_Q).mean()
        
        print(f"The measured angle of demodulated I and Q is {theta1} rad at {CarrierTones[0]/1e3} kHz and {theta2} rad at {CarrierTones[1]/1e3} kHz, the calib angle is {angle} rad at {CarrierTones[0]/1e3} kHz and {angle_tone2} rad at {CarrierTones[1]/1e3} kHz")
        
        
        No_scale = False
        # angle = -theta1/2
        # if No_scale:
        #     data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD_nonescale(
        #         f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['I_onres%d'%i][()],
        #         f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['Q_onres%d'%i][()],
        #         angle)
        # else:
        #     data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(
        #         f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['I_onres%d'%i][()],
        #         f[key_BBtemp][key_temp][key_pwr]['res_scan']['noise']['Q_onres%d'%i][()],
        #         angle, dIQ_vs_df, resfit_dict['fres'])
        # f_data, Sdff_onres_freq = sig.welch(data_noise_onres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
        # _, Sdff_onres_diss = sig.welch(data_noise_onres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
        
        if No_scale:
            data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD_nonescale(tone1_I, tone1_Q, angle)
        else:
            data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(tone1_I, tone1_Q, angle, dIQ_vs_df, resfit_dict['fres'])
        f_data, Sdff_onres_freq = sig.welch(data_noise_onres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
        _, Sdff_onres_diss = sig.welch(data_noise_onres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
        # data_noise_onres_freq, data_noise_onres_diss = rotate_noise_IQ2FD(tone1_I, tone1_Q, angle, dIQ_vs_df, resfit_dict['fres'])
        # f_data, Sdff_onres_freq = sig.welch(data_noise_onres_freq, fs=fs/2, window='hann', nperseg=nperseg, return_onesided = True)
        # _, Sdff_onres_diss = sig.welch(data_noise_onres_diss, fs=fs/2, window='hann', nperseg=nperseg, return_onesided = True)
        
        # LP_cutoff=100000

        
        # angle = 4.229817094826354
        # dIQ_vs_df = 6.848923982320005e-5

        if No_scale:
            data_noise_offres_freq, data_noise_offres_diss = rotate_noise_IQ2FD_nonescale(tone2_I, tone2_Q, angle_tone2)
        else:
            data_noise_offres_freq, data_noise_offres_diss = rotate_noise_IQ2FD(tone2_I, tone2_Q, angle_tone2, dIQ_vs_df_tone2, resfit_dict_tone2['fres'])
        f_data, Sdff_offres_freq = sig.welch(data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
        _, Sdff_offres_diss = sig.welch(data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
        # # data_noise_offres_freq, data_noise_offres_diss = rotate_noise_IQ2FD(tone1_Ioff, tone1_Qoff, angle, dIQ_vs_df, resfit_dict['fres']+main_dict['df_onoff_res'])
        # # f_data, Sdff_offres_freq = sig.welch(data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
        # # _, Sdff_offres_diss = sig.welch(data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
        
        
        #trying to clean noise
        CleanedNoise_Freq_Onres, alpha_f, alpha_est, f_noise = extract_noise(data_noise_onres_freq, data_noise_offres_freq, fs, 50, 2)        
        f_data, Sdff_CleanedNoise_Onres_freq = sig.welch(CleanedNoise_Freq_Onres, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)

        CleanedNoise_Diss_Onres, _, _, _ = extract_noise(data_noise_onres_diss, data_noise_offres_diss, fs, 50, 2)
        f_data, Sdff_CleanedNoise_Onres_diss = sig.welch(CleanedNoise_Diss_Onres, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
        
        CleanedNoise_Freq_Offres, _, _, _ = extract_noise(data_noise_offres_freq, data_noise_onres_freq, fs, 50, 2)
        f_data, Sdff_CleanedNoise_Offres_freq = sig.welch(CleanedNoise_Freq_Offres, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)

        CleanedNoise_Diss_Offres, _, _, _  = extract_noise(data_noise_offres_diss, data_noise_onres_diss, fs, 50, 2)
        f_data, Sdff_CleanedNoise_Offres_diss = sig.welch(CleanedNoise_Diss_Offres, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
        
        Y_limit = [1e-20, 2e-15]
        fig = plt.figure()
        plt.loglog(f_noise, alpha_f, '.', label='Alpha bin by bin')
        plt.axhline(alpha_est, color='r', label='Sum of FFT bins')
        plt.xlabel("Frequency (Hz)", fontsize=14)
        plt.ylabel(r"Alpha ($\alpha$)", fontsize=14)
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        
        Sdf_dir = '/'.join(file_path.split('/')[:-2]) + '/Plots/Sdff/'
        filename = Sdf_dir + 'Alpha'
        print('Saving the Plot Alpha')
        fig.savefig(filename + '.png', dpi=100)
        
        
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
        fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
        axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_onres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', label = 'onres: freq')
        axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_CleanedNoise_Onres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha=0.5, color='b', label = 'onres: freq cleaned')
        axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_onres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', label = 'onres: diss')
        axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_CleanedNoise_Onres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha=0.5, color='r', label = 'onres: diss cleaned')
        axs[0].tick_params(axis='y', labelsize=14)
        axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
        axs[0].tick_params(axis='x', labelsize=14)
        axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
        axs[0].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
        axs[0].relim()
        axs[0].set_ylim(Y_limit)
        axs[0].autoscale_view()  
        figtitle = '%s:  %dmK  fitres %sMHz'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), meas_dict['freq_center_str'])
        fig.text(0.5, 0.95, figtitle, fontsize=12, horizontalalignment='center', verticalalignment='top')
        
        axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', label = 'offres: freq')
        axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_CleanedNoise_Offres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha=0.5, color='b', label = 'offres: freq cleaned')
        axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', label = 'offres: diss')
        axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_CleanedNoise_Offres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha=0.5, color='r', label = 'offres: diss cleaned')
        axs[1].tick_params(axis='y', labelsize=14)
        axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
        axs[1].tick_params(axis='x', labelsize=14)
        axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
        axs[1].set_ylabel(r'Sdf/f  [$Hz^{-1}$]', fontsize = 16)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
        axs[1].relim()
        axs[1].set_ylim(Y_limit)
        axs[1].autoscale_view()
        
        Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
        filename = Sdf_dir + 'TwoTones_Sdff'
        print('Saving the Plot TwoTones_Sdff')
        fig.savefig(filename + '.pdf', dpi=100)
        
        
        #Coherence calculation
        # Compute coherence
        C_freq, C_xy_freq = coherence(data_noise_onres_freq, data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg)
        C_freq, C_xy_diss = coherence(data_noise_onres_diss, data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg)
        
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
        fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
        axs[0].loglog(C_freq[:int(LP_cutoff*S_time/nfactor)], C_xy_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', label = 'Coherence: freq')
        axs[0].loglog(C_freq[:int(LP_cutoff*S_time/nfactor)], C_xy_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', label = 'Coherence: diss')
        axs[0].tick_params(axis='y', labelsize=14)
        axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
        axs[0].tick_params(axis='x', labelsize=14)
        axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
        axs[0].set_ylabel(r'Coherence', fontsize = 16)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
        axs[0].relim()
        # axs[0].set_ylim([1e-20, 1e-15])
        axs[0].autoscale_view()  
        figtitle = '%s:  %dmK  fitres %sMHz'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), meas_dict['freq_center_str'])
        fig.text(0.5, 0.95, figtitle, fontsize=12, horizontalalignment='center', verticalalignment='top')
        
        #Cross PSD (CSD)
        Sxy_freq, S_xy_freq = csd(data_noise_onres_freq, data_noise_offres_freq, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)
        Sxy_freq, S_xy_diss = csd(data_noise_onres_diss, data_noise_offres_diss, fs=fs, window='hann', nperseg=nperseg, return_onesided = True)        
        
        axs[1].loglog(Sxy_freq[:int(LP_cutoff*S_time/nfactor)], S_xy_freq [:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', label = 'CSD: freq')
        axs[1].loglog(Sxy_freq[:int(LP_cutoff*S_time/nfactor)], S_xy_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', label = 'CSD: diss')
        axs[1].tick_params(axis='y', labelsize=14)
        axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
        axs[1].tick_params(axis='x', labelsize=14)
        axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
        axs[1].set_ylabel(r'Sxy  [$Hz^{-1}$ ?]', fontsize = 16)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
        axs[1].relim()
        # axs[1].set_ylim([1e-20, 1e-15])
        axs[1].autoscale_view()
        
        Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
        filename = Sdf_dir + 'Coherence_CrossPSD'
        print('Saving the Plot Coherence_CrossPSD')
        fig.savefig(filename + '.png', dpi=100)
        
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
        fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
        
        axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_onres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', label = 'tones1: freq')
        axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', label = 'tones2: freq')
        axs[0].loglog(Sxy_freq[:int(LP_cutoff*S_time/nfactor)], S_xy_freq [:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C2', label = 'CSD: freq')
        axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_CleanedNoise_Onres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha=0.5, color='b', label = 'tones1: freq cleaned')
        axs[0].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_CleanedNoise_Offres_freq[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha=0.5, color='r', label = 'tones2: freq cleaned')
        axs[0].tick_params(axis='y', labelsize=14)
        axs[0].grid(visible=True, which='both', color='0.75', linestyle='-')
        axs[0].tick_params(axis='x', labelsize=14)
        axs[0].set_xlabel('Frequency  [Hz]', fontsize = 16)
        axs[0].set_ylabel(r'Frequency Sdf/f  [$Hz^{-1}$]', fontsize = 16)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
        axs[0].relim()
        axs[0].set_ylim(Y_limit)
        axs[0].autoscale_view()  
        figtitle = '%s:  %dmK  fitres %sMHz'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), meas_dict['freq_center_str'])
        fig.text(0.5, 0.95, figtitle, fontsize=12, horizontalalignment='center', verticalalignment='top')

        axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_onres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C0', label = 'onres: diss')
        axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_offres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C1', label = 'tones2: diss')
        axs[1].loglog(Sxy_freq[:int(LP_cutoff*S_time/nfactor)], S_xy_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, color='C2', label = 'CSD: diss')
        axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_CleanedNoise_Onres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha=0.5, color='b', label = 'tones1: diss cleaned')
        axs[1].loglog(f_data[:int(LP_cutoff*S_time/nfactor)], Sdff_CleanedNoise_Offres_diss[:int(LP_cutoff*S_time/nfactor)], linewidth=2, alpha=0.5, color='r', label = 'tones2: diss cleaned')
        axs[1].tick_params(axis='y', labelsize=14)
        axs[1].grid(visible=True, which='both', color='0.75', linestyle='-')
        axs[1].tick_params(axis='x', labelsize=14)
        axs[1].set_xlabel('Frequency  [Hz]', fontsize = 16)
        axs[1].set_ylabel(r'Dissipation Sdf/f  [$Hz^{-1}$]', fontsize = 16)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
        axs[1].relim()
        axs[1].set_ylim(Y_limit)
        axs[1].autoscale_view()        
        
        Sdf_dir = '/'.join(file_path.split('/')[:-2])+'/Plots/Sdff/'
        filename = Sdf_dir + 'Freq_and_Diss'
        print('Saving the Plo Freq_and_Diss')
        fig.savefig(filename + '.png', dpi=100)
        


#%%
dir_path = 'D:/noise_data/IQ_data/'
# MB_filename = 'D:/YellowCryostatData/Be210504p2bl/FPtemp_sweep/20220704_21h40/Data/data_h5/Be210504p2bl_dark_merged_processed.h5'
# device_allres = np.array([172.902e6, 177.499e6, 208.673e6, 209.238e6, 211.933e6, 215.182e6, 244.780e6, 246.228e6, 247.845e6, 251.390e6, 254.826e6, 258.301e6, 263.124e6, 264.553e6, 266.425e6, 271.605e6, 312.07e6, 312.4e6, 313.780e6, 315.08e6, 315.444e6, 319e6, 319.4e6, 322.574e6, 324.892e6, 325.671e6, 326.768e6, 327.643e6, 331.211e6, 332.878e6, 368.617e6, 371.640e6, 374.184e6, 375.591e6, 377.396e6, 381.510e6, 382.089e6, 384.974e6, 386.939e6, 391.857e6, 392.874e6, 394.665e6, 398.146e6, 399.593e6, 404.976e6, 405.601e6, 494.132e6, 642.762e6])

# MB_filename = 'D:/YellowCryostatData/Be231102d2_AR_BS/Dark/20240214_11h46/Data/data_h5/Be231102d2_AR_BS_dark_processed.h5' 
# device_allres = np.array([213.08e6, 214.80e6, 247.41e6, 255.83e6, 256.86e6, 260.12e6, 263.86e6, 265.55e6, 269.67e6, 270.30e6, 272.67e6, 274.71e6, 276.38e6, 278.96e6, 308.52e6, 312.88e6, 313.15e6, 314.00e6, 314.39e6, 318.56e6, 322.41e6, 323.72e6, 326.36e6, 328.32e6, 328.71e6, 335.79e6, 340.65e6, 368.18e6, 373.47e6, 375.80e6, 375.95e6, 377.84e6, 384.08e6, 386.53e6, 386.87e6, 387.22e6, 393.64e6, 397.18e6, 401.25e6, 404.14e6, 409.10e6, 417.63e6, 504.12e6, 560.88e6, 596.45e6, 611.96e6, 628.62e6])

# MB_filename = 'D:/YellowCryostatData/Be210504p2b1/Dark/20240215_20h50/Data/data_h5/Be210504p2b1_dark_processed.h5' 
# MB_filename = 'D:/YellowCryostatData/Be231102d1_AR_NoBS/Dark/20240215_09h58/Data/data_h5/Be231102d1_AR_NoBS_dark_processed.h5' 
# MB_filename = 'D:/YellowCryostatData/cf221001_NoAR_NoBS/Dark/20240216_17h03/Data/data_h5/cf221001_NoAR_NoBS_dark_processed.h5' 

# res_fileindex = 0
# filenames =  [] 
# main_dict, meas_dict, resfit_dict, calib_dict, data_dict = load_Sdff(dir_path, device_allres)
# resfit_dict = reso_LPfilter(resfit_dict)
# filenames.append(plot_scan_params(main_dict, meas_dict, data_dict, saveplots=True))
# # filenames.append(glob.glob(main_dict['plot_dir'] + 'freq_sweep/'+'*.pdf')[res_fileindex].replace('\\','/'))
# # filenames.append(glob.glob(main_dict['plot_dir'] + 'fit_res/'+'*.pdf')[res_fileindex].replace('\\','/'))
# # filenames.append(glob.glob(main_dict['plot_dir'] + 'calib_rotation/'+'*.pdf')[res_fileindex].replace('\\','/'))
# filenames=plot_noise_time_blob(main_dict, meas_dict, resfit_dict, calib_dict, data_dict, filenames, saveplots=True)
# filenames=plot_avg_Sdff_data(data_dict, resfit_dict, main_dict, meas_dict, filenames, dec_fact=100, ylims=[1e-21, 1e-14], saveplots=True)
# # plot_all_noise_psd(main_dict, meas_dict, resfit_dict, data_dict, saveplots=True)
# data_dict = calc_SNqp(data_dict, resfit_dict, MB_filename)
# fit_dict = get_tauqp_Nqp(data_dict, resfit_dict, f_range=(3e2, 5e4))
# data_dict = fit_SNqp(data_dict, resfit_dict, fit_dict)
# data_dict = get_NEP(data_dict, fit_dict, resfit_dict, V_KIDs, MB_filename)
# data_dict = get_E(data_dict, resfit_dict)
# filenames=plot_SNqp_NEP(data_dict, fit_dict, resfit_dict, main_dict, meas_dict, V_KIDs, MB_filename, filenames, saveplots=True)
# save_data(main_dict, data_dict, fit_dict)
# # save_pdf_report(main_dict['plot_dir'], meas_dict, filenames)

# ResFreq=get_allresFreq_data(dir_path)

# dir_path = 'D:/noise_data/IQ_data/Be231102d2/20241021_183518-OneTone-0dBm-500kHz'

#%%

# if __name__ == "__main__":

#     main_dict, meas_dict, resfit_dict, calib_dict, data_dict = load_Sdff(dir_path, device_allres)

#     plot_all_noise_psd(main_dict, meas_dict, resfit_dict, data_dict, saveplots=False)

#     check_data(dir_path)
