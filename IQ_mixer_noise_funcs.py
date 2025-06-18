# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:30:55 2023

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
from scipy.optimize import curve_fit #yay \o/
sys.path.append('D:/Python_scripts/scraps/')
import scraps as scr
import scipy.signal as sig
import lmfit as lf
import scipy.special as sp
from Hardware_Control import set_MiniCAtten, GetBoxAtten
from Hardware_Control import setBoxPortSwitch
import scipy.special as sp

IQ_calibfileDC_path = 'D:/noise_data/IQ_calib/20240325_133843_DC_gain20_RFm16/h5_data/IQ_calib_fit.h5'
IQ_calibfileAC_path = 'D:/noise_data/IQ_calib/20240325_144350_AC_gain200_RFm16/h5_data/IQ_calib_fit.h5'

kb_eV = 8.6173423e-5 # [eV/K] Boltzmann cte
h_eV = 4.135667696e-15 # [eV.s] Planck cte
hbar_eV = h_eV/(2 * np.pi)
N0 = 1.71e10 # [eV/um^3] Single spin electron density of states at the Fermi level (S. Siegel thesis, p44)

mydpi = 100 # Resolution for displayed plots



def init_daq(meas_dict):
    '''
    Initialize the DAQ NI 9775
    3 channels (over 4) are being used (chan0: I, chan1: Q, chan2: modulation signal)
    The modulation signal is comes from the signal generator and it give the frequency variation over sweeps and frequency modulation
    The 4 modes (freq_sweep, FM, calib, noise) have different values for DAQ frequency, total Npts, and Npts/period
    '''
    chan0_str = "cDAQ1Mod1/ai0"
    chan1_str = "cDAQ1Mod1/ai1"
    chan2_str = "cDAQ1Mod1/ai2"
    print('\n --- DAQ initialization ---')
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(chan0_str)
    task.ai_channels.add_ai_voltage_chan(chan1_str)
    task.ai_channels.add_ai_voltage_chan(chan2_str)
    print('Channels %s, %s and %s added'%(chan0_str, chan1_str, chan2_str))
    task.timing.adc_sample_high_res()
    print('Timing mode set to High res')
    
    if meas_dict['mode'] == 'freq_sweep':
        meas_dict['Npts_period'] = int(1e5) # Num of points per sweep (arbitrary, change if necessary)
        meas_dict['DAQ_freq'] = int(meas_dict['rate'] * meas_dict['Npts_period']) # Hz, DAQ acquisition frequency
        meas_dict['DAQ_Npts'] = int(2 * meas_dict['Npts_period']) # to capture the sweep over 1 period we need to measure a total of 2 periods

    elif meas_dict['mode'] == 'FM':
        meas_dict['Npts_period'] = int(1e4) # Num of points per period (arbitrary, change if necessary)
        meas_dict['DAQ_freq'] = int(meas_dict['rate'] * meas_dict['Npts_period']) # Hz, DAQ acquisition frequency
        meas_dict['DAQ_Npts'] = int(2 * meas_dict['Npts_period']) # to capture the sweep over 1 period we need to measure a total of 2 periods

    elif meas_dict['mode'] == 'calib':
        meas_dict['Npts_period'] = int(1e4) # Num of points per period (arbitrary, change if necessary)
        meas_dict['DAQ_freq'] = int(meas_dict['rate'] * meas_dict['Npts_period']) # Hz, DAQ acquisition frequency
        meas_dict['DAQ_Npts'] = int(21 * meas_dict['Npts_period']) # to capture the calib over 20 periods we need to measure a total of 21 periods

    elif meas_dict['mode'] == 'noise':
        # meas_dict['DAQ_freq'] = int(2e6) # Hz, DAQ acquisition frequency #07/17/2024 change the value to 2e6 for two tone setup
        # meas_dict['DAQ_Npts'] = int(2e6) # to capture noise over 10 s #04/28/2024 change the value to 1e6(1s) from 10e6 #06/04/2024 change the value to 30e6(30s) from 10e6
        #for twotone test
        task.close()
        task = nidaqmx.Task()
        task.ai_channels.add_ai_voltage_chan(chan0_str)
        task.ai_channels.add_ai_voltage_chan(chan1_str)
        print('Channels %s and %s added'%(chan0_str, chan1_str))
        task.timing.adc_sample_high_res()
        print('Timing mode set to High res')
        # task.timing.adc_sample_high_speed()
        meas_dict['DAQ_freq'] = int((20/12)*1e6) # Hz, DAQ acquisition frequency #07/17/2024 change the value to 2e6 for two tone setup
        meas_dict['DAQ_Npts'] = int(meas_dict['DAQ_freq']*100) # to capture noise over 10 s #04/28/2024 change the value to 1e6(1s) from 10e6 #06/04/2024 change the value to 30e6(30s) from 10e6

    else: 
        print('Mode not valid (must be either freq_sweep, FM, or noise). Abort')
        return task, meas_dict
    
    meas_dict['DAQ_acq_time'] = meas_dict['DAQ_Npts'] / meas_dict['DAQ_freq']
    
    args = (meas_dict['mode'], meas_dict['DAQ_freq']/1e3, meas_dict['DAQ_Npts'], meas_dict['DAQ_acq_time'])
    scan_str = '''\n--------------------------------------------------
    --- DAQ NI 9775 settings --- 
        Measurement mode: {:s} 
        DAQ acquisition rate: {:.1f} kHz
        DAQ number of acquired data points: {:.2e} 
        Measurement duration: {:.3f} s
        
    '''.format(*args)    
    print(scan_str)
    return task, meas_dict


def daq_get_data(main_dict, meas_dict, task):
    task.timing.cfg_samp_clk_timing(meas_dict['DAQ_freq'], sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=meas_dict['DAQ_Npts'])
    # print('DAQ sampling frequency = %.1f kHz'%(meas_dict['DAQ_freq']/1e3))
    # print('DAQ Npts = %d'%meas_dict['DAQ_Npts'])
    # print('DAQ measurement duration = %.3f s'%(meas_dict['DAQ_Npts']/meas_dict['DAQ_freq']))
    print('DAQ measurement started ...')
    daq_data = np.array(task.read(number_of_samples_per_channel=meas_dict['DAQ_Npts'], timeout=nidaqmx.constants.WAIT_INFINITELY))
    if main_dict['IQ_mixer'] == 'SigaTek QD54A10':
        daq_data[1] = daq_data[1]*-1 # we take the inverse of Q data, because the resonance fit fails if we don't
    elif main_dict['IQ_mixer'] == 'TSC AD0460B':
        daq_data[1] = daq_data[1]*1 # we don't take the inverse of Q data, nothing happens
    else:
        print('Unknown mixer name, exit')
        return
    return daq_data
    

def get_atten_SG2mixer(freq, board="LF"):
    '''
    freq in Hz
    Attenuation between the signal generator and the mixer, for both boards.
    LF board is the low frequency board (from 10 MHz to 1500 MHz) and 
    HF board is the high frequency board (from 1400 MHz to 2 GHz)
    Measurement from 12/04/2023, accurate within +/- 0.3 dB
    Updated on 03/22/2024 with fit from 300 Mhz to 2.2 GHz for Hfreq board
    '''
    p_LF = np.poly1d([-3.717e-45, 1.523e-35, -2.248e-26, 1.430e-17, -4.218e-09, -3.042])
    # p_HF = np.poly1d([ 9.476e-44, -7.742e-34,  2.512e-24, -4.043e-15, 3.229e-06, -1.027e+03])
    p_HF = np.poly1d([-1.828e-54, 1.446e-44, -4.561e-35, 7.298e-26, -6.202e-17, 2.617e-08, -7.512])  
    
    LF_range = (10e6, 1500e6)
    HF_range = (350e6, 2.1e9)
    if board == "LF":
        if (np.asarray(freq) >= LF_range[0]).all() and (np.asarray(freq) <= LF_range[1]).all():
            S21_atten_dB = p_LF(freq)
        else:
            print("Wrong frequency range for LF")
            return
    elif board == "HF":
        if (np.asarray(freq) >= HF_range[0]).all() and (np.asarray(freq) <= HF_range[1]).all():
            S21_atten_dB = p_HF(freq)
        else:
            print("Wrong frequency range for HF")
            return
    else:
        print("Wrong board identifier (must be LF or HF)")
        return
    return S21_atten_dB



def get_atten_RF2mixer(freq, board="LF"):
    '''
    freq in Hz
    Attenuation between the RF input of the IQ mixer board and the mixer, for both boards.
    LF board is the low frequency board (from 10 MHz to 1500 MHz) and 
    HF board is the high frequency board (from 1400 MHz to 2 GHz)
    Measurement from 12/04/2023, accurate within +/- 0.3 dB
    Updated on 03/22/2024 with fit from 300 Mhz to 2.2 GHz for Hfreq board
    '''
    p_LF = np.poly1d([-3.717e-45, 1.523e-35, -2.248e-26, 1.430e-17, -4.218e-09, -3.042])
    # p_HF = np.poly1d([ 9.476e-44, -7.742e-34,  2.512e-24, -4.043e-15, 3.229e-06, -1.027e+03])
    p_HF = np.poly1d([-1.828e-54, 1.446e-44, -4.561e-35, 7.298e-26, -6.202e-17, 2.617e-08, -7.512])  

    LF_range = (10e6, 1500e6)
    HF_range = (350e6, 2.1e9)
    if board == "LF":
        if (np.asarray(freq) >= LF_range[0]).all() and (np.asarray(freq) <= LF_range[1]).all():
            S21_atten_dB = p_LF(freq)
        else:
            print("Wrong frequency range for LF")
            return
    elif board == "HF":
        if (np.asarray(freq) >= HF_range[0]).all() and (np.asarray(freq) <= HF_range[1]).all():
            S21_atten_dB = p_HF(freq)
        else:
            print("Wrong frequency range for HF")
            return
    else:
        print("Wrong board identifier (must be LF or HF)")
        return
    return S21_atten_dB


def get_atten_SG2LOout(freq, board="LF"):
    '''
    freq in Hz
    Attenuation between the signal generator and the LO output of the mixer board, for both boards.
    LF board is the low frequency board (from 10 MHz to 1500 MHz) and 
    HF board is the high frequency board (from 400 MHz to 2 GHz)
    Measurement from 12/04/2023, accurate within +/- 0.3 dB
    Updated on 03/22/2024 with fit from 300 Mhz to 2.2 GHz for Hfreq board
    '''
    p_LF = np.poly1d([ 9.898e-47, -5.527e-38, -1.011e-27,  1.601e-18, -1.820e-09, -6.489e+00])
    # p_HF = np.poly1d([ 4.819e-44, -3.902e-34,  1.254e-24, -1.999e-15, 1.581e-06, -5.027e+02])
    p_HF = np.poly1d([ -1.193e-54, 9.489e-45, -3.025e-35, 4.93e-26, -4.309e-17, 1.871e-08, -9.773])

    LF_range = (10e6, 1500e6)
    HF_range = (350e6, 2.1e9)
    if board == "LF":
        if (np.asarray(freq) >= LF_range[0]).all() and (np.asarray(freq) <= LF_range[1]).all():
            S21_atten_dB = p_LF(freq)
        else:
            print("Wrong frequency range for LF")
            return
    elif board == "HF":
        if (np.asarray(freq) >= HF_range[0]).all() and (np.asarray(freq) <= HF_range[1]).all():
            S21_atten_dB = p_HF(freq)
        else:
            print("Wrong frequency range for HF")
            return
    else:
        print("Wrong board identifier (must be LF or HF)")
        return
    return S21_atten_dB


def get_atten_insidecryo(freq):
    ''' 
    The attenuation measured between the input of the yellow cryostat and the device when the cryostat is cold:
    |S21(f)| = -8.75e-10 * f - 41.5 
    With |S21| in dB and f in Hz.  
    It includes the losses of the coax cables and the attenuators
    (measurements from 10/2018 by F. Defrance)
    '''
    S21_atten_dB = -8.75e-10 * freq - 41.5 
    return S21_atten_dB


def get_atten_SMASwitch_and_LongSMACable(freq):
    '''
    freq in Hz
    The measured attenuation of 1 long (2m) SMA cable + SMA switch used at the input and ouput of the yellow cryostat.
    Measurement from March/22/2024, accurate within +/- 0.3 dB
    '''
    p = np.poly1d([7.071e-57, -9.938e-47, 5.313e-37, -1.357e-27, 1.749e-18, -1.666e-09, -1.04])
    freq_range = (10e6, 4000e6)

    if (np.asarray(freq) >= freq_range[0]).all() and (np.asarray(freq) <= freq_range[1]).all():
        S21_atten_dB = p(freq)
    else:
        print("Wrong board identifier")
        return
    return S21_atten_dB


def get_warm_LNA_gain(freq):
    '''
    freq in Hz
    Gain of the warm LNAs box.
    Measurement from 12/04/2023, accurate within +/- 0.3 dB
    New Measurement from March/13/2023, accurate within +/- 0.3 dB
    '''
    # p_gain = np.poly1d([ 2.99725611e-54, -2.02297978e-44,  5.37875590e-35, -7.16666575e-26,
    #     5.08134897e-17, -2.28232755e-08,  4.52662395e+01])
    # freq_range = (50e6, 2000e6)
    p_gain = np.poly1d([ -6.061e-55, 5.121e-45, -1.576e-35, 2.124e-26, -8.416e-18, -1.467e-08, 69.57])
    freq_range = (30e6, 3500e6)

    if (np.asarray(freq) >= freq_range[0]).all() and (np.asarray(freq) <= freq_range[1]).all():
        S21_gain_dB = p_gain(freq)
    else:
        print("Wrong frequency range for warm LNAs")
        return
    return S21_gain_dB


def get_cryo_LNA_gain(cryo_chan, freq):
    '''
    freq in Hz
    Returns the gain of the cryo LNA in dB.
    '''
    p_gain_chan1 = np.poly1d([30]) # to be measured
    p_gain_chan2 = np.poly1d([30]) # to be measured
    p_gain_chan3 = np.poly1d([32]) # to be measured
    p_gain_chan4 = np.poly1d([4.96980979e-37, -4.95253827e-29, -6.06218644e-18,  6.19708622e-09, 3.05200090e+01])
    
    freq_range_chan1 = (150e6, 2000e6) # LNC0.6_2A #temporarily change the value to (150e6, 2000e6) from (600e6, 2000e6)
    freq_range_chan2 = (600e6, 2000e6) # LNC0.6_2A 
    freq_range_chan3 = (10e6, 2000e6) # ASU 2GHz
    freq_range_chan4 = (10e6, 2000e6) # CMT CITLF2

    if cryo_chan == 1:
        if (np.asarray(freq) >= freq_range_chan1[0]).all() and (np.asarray(freq) <= freq_range_chan1[1]).all():
            cryo_LNA_gain = cryo_LNA_gain = p_gain_chan1(freq)
        else:
            print("Wrong frequency range for cryo LNA")
            return
    elif cryo_chan == 2:
        if (np.asarray(freq) >= freq_range_chan2[0]).all() and (np.asarray(freq) <= freq_range_chan2[1]).all():
            cryo_LNA_gain = cryo_LNA_gain = p_gain_chan2(freq)
        else:
            print("Wrong frequency range for cryo LNA")
            return
    elif cryo_chan == 3:
        if (np.asarray(freq) >= freq_range_chan3[0]).all() and (np.asarray(freq) <= freq_range_chan3[1]).all():
            cryo_LNA_gain = cryo_LNA_gain = p_gain_chan3(freq)
        else:
            print("Wrong frequency range for cryo LNA")
            return
    elif cryo_chan == 4:
        if (np.asarray(freq) >= freq_range_chan4[0]).all() and (np.asarray(freq) <= freq_range_chan4[1]).all():
            cryo_LNA_gain = cryo_LNA_gain = p_gain_chan4(freq)
        else:
            print("Wrong frequency range for cryo LNA")
            return
    else:
        print("Wrong cryo_chan for cryo LNA (must be 1, 2, 3, or 4)")
        return
    return cryo_LNA_gain


def get_gain_afterdevice(main_dict, freq):
    '''
    The gain of the LNA for channels 1, 2, 3 is 32 dB, according to the datasheet
    The variable attenuation after the device is set by a variable Minicircuit attenuator.
    The fit is accurate up to 2 GHz.
    This calculation does not consider losses between the cryogenic LNA and the output of the cryostat, but we expect 
    these to be small.
    This function calculates the total gain after the device without considering the variable attenuator. Therefore we are
    using this function to calculate the attenuation of the variable attenuator to adjust the power coming back to the mixer.
    '''

    cryo_LNA_gain = get_cryo_LNA_gain(main_dict['cryo_chan'], freq)
    warm_LNA_gain = get_warm_LNA_gain(freq)
    atten_cryo2attenbox = get_atten_SMASwitch_and_LongSMACable(freq)
    atten_RF2mixer = get_atten_RF2mixer(freq, board=main_dict['board'])
    print(cryo_LNA_gain, warm_LNA_gain, atten_cryo2attenbox, atten_RF2mixer)
    S21_gain_dB = warm_LNA_gain + cryo_LNA_gain + atten_RF2mixer + atten_cryo2attenbox
    return S21_gain_dB

    
def define_freq_sweep_bands(meas_dict):
    '''
    Signal generator SG382 cannot continuously sweep from 0 to 2 GHz, so we need to split the sweep into several bands
    This function checks if the sweep requirements are compatible with SG382 specs and defines the frequency bands
    that need to be swept.
    '''
    
    freq_start = meas_dict['freq_start']
    freq_stop = meas_dict['freq_stop']

    # The signal generator can only continuously sweep the frequency within specific bands:
    bandfreq = np.array([62, 125, 250, 500, 1000, 2000])*1e6 # in Hz
    if freq_start > freq_stop:
        print('freq_start > freq_stop, the 2 values will be inverted')
        freq_start_tmp = int(freq_stop)
        freq_stop = int(freq_start)
        freq_start = freq_start_tmp
    if freq_stop > bandfreq[-1]:
        print('The max frequency of the sweep is above the limit (2 GHz)')
        print('Max freq set to 2 GHz')
        freq_stop = bandfreq[-1]
    if freq_start < 0:
        print('The min frequency of the sweep is below 0 GHz')
        print('Min freq set to 0 GHz')
        freq_start = 0
        
    findFspanL = freq_start - bandfreq
    findFspanH = freq_stop - bandfreq
    L_bandfreq_i = np.where(findFspanL <= 0)[0][0] # index of the low frequency band
    H_bandfreq_i = np.where(findFspanH <= 0)[0][0] # index of the high frequency band
    N_bands = H_bandfreq_i - L_bandfreq_i + 1 # Number of frequency bands to sweep
    freq_start_array = np.zeros(N_bands)
    freq_stop_array = np.zeros(N_bands)
    for i in range(N_bands):
        i_band = L_bandfreq_i + i
        if i == 0:
            freq_L = freq_start
        else: 
            freq_L = bandfreq[i_band-1]
        if i == N_bands-1:
            freq_H = freq_stop
        else:
            freq_H = bandfreq[i_band]
        
        freq_start_array[i] = int(freq_L)
        freq_stop_array[i] = int(freq_H)
    meas_dict['freq_start_array'] = freq_start_array
    meas_dict['freq_stop_array'] = freq_stop_array
    meas_dict['N_bands'] = N_bands
    return meas_dict

    
def init_measurement(main_dict):
    print('\nSet SMA switch to channel %d'%main_dict['cryo_chan'])
    setSMASwitch.SetSMASwitch(main_dict['cryo_chan'])
    
    if main_dict['IQ_mixer'] == 'SigaTek QD54A10':
        setBoxPortSwitch.SetBoxSwitch(1)
        main_dict['LO_pwr'] = 11 # dBm: for Sigatek QD54A10 mixer, the input LO power must be between 10 and 13 dBm
        main_dict['max_RFpwr_mixer'] = -5 # dBm: for Sigatek QD54A10 mixer, 
        # the max input RF power is +5 dBm but we would like to avoid any RF power above -10 dBm due to uncertainty on power levels
        main_dict['RFpwr_req@mixer'] = 40 # dBm: Desired power level for mixer RF input #-10
        main_dict['board'] = 'LF'
    elif main_dict['IQ_mixer'] == 'TSC AD0460B':
        setBoxPortSwitch.SetBoxSwitch(2)
        main_dict['LO_pwr'] = 0 # dBm: for TSC AD0460B mixer, the input LO power must be between -3 and +3 dBm
        main_dict['max_RFpwr_mixer'] = 0 # dBm: for TSC AD0460B mixer, 
        # the max input RF power is +15 dBm but we would like to avoid any RF power above 0 dBm due to uncertainty on power levels
        main_dict['RFpwr_req@mixer'] = -10 # dBm: Desired power level for mixer RF input
        main_dict['board'] = 'HF'
    else:
        print('Wrong IQ mixer model, abort')
        return
    
    for key in ['pwr_sweep_bool', 'FPtemp_sweep_bool', 'source_sweep_bool']:
        main_dict[key] = False
    
    main_dict['Nres'] = len(main_dict['freq_center'])
    main_dict['time_yyyymmdd'] = time.strftime('%Y%m%d')
    main_dict['time_hhmmss'] = time.strftime('%H%M%S')
    main_dict['FPtemp_ini'] = float(GRT.GetTemp())    
    args = (main_dict['time_yyyymmdd']  + main_dict['time_hhmmss'], main_dict['dev_name'], 
    main_dict['cryo_chan'], main_dict['FPtemp_ini']*1e3)
    scan_str = '''\n--------------------------------------------------
    --- IQ mixer measurement --- 
    Measurement parameters:
        Date: {:s}
        Device name: {:s}
        Cryostat channel: {:d} (channel used for the device: [1 - 4])
        Current focal plane temperature: {:.2f} mK
    '''.format(*args)
    
    if len(main_dict['source']) > 1 :
        main_dict['source_sweep_bool'] = True
        scan_str += '    Source sweep: %s\n'%(main_dict['source'])
    else:
        scan_str += '    Source: %s\n'%(main_dict['source'])
        
    if main_dict['pwr_sweep'] is not None:
        main_dict['pwr_sweep'] = np.sort(main_dict['pwr_sweep'])
        # meas_dict['req_pwr@device'] = np.array([a + main_dict['pwr_sweep'] for a in meas_dict['req_pwr@device']])
        main_dict['pwr_sweep_bool'] = True
        scan_str += '        Readout power sweep (variation in dB): %s\n'%(main_dict['pwr_sweep'])
    else: 
        main_dict['pwr_sweep'] = [0]
    
    if main_dict['FP_temp'] is not None:
        main_dict['FP_temp'] = np.sort(main_dict['FP_temp'])
        main_dict['FPtemp_sweep_bool'] = True
        scan_str += '        Focal plane temperature sweep: %s mK\n'%(main_dict['FP_temp']*1e3)
    else:
        main_dict['FP_temp'] = [main_dict['FPtemp_ini']]
        
    print(scan_str)
    
    return main_dict


def define_atten_and_pwr(main_dict):
    
    freq_arr = main_dict['freq_center']
    max_atten_in, max_atten_out = GetBoxAtten.get_atten_SwitchBox(freq_arr, -90, -90) # dBm, maximum attenuation of the variable attenuators
    min_atten_in, min_atten_out = GetBoxAtten.get_atten_SwitchBox(freq_arr, 0, 0)  # dBm, minimum attenuation of the variable attenuators    
    min_atten_out = np.ones(len(min_atten_out)) * -10 # for safety until we know the power levels
    
    
    main_dict['SG382_pwr'] = np.mean(main_dict['LO_pwr'] - get_atten_SG2mixer(freq_arr, board=main_dict['board']))
    atten_attenbox2cryo = get_atten_SMASwitch_and_LongSMACable(freq_arr) # attenuation between attenuator box and cryostat (long SMA cable + SMA switch)
    # Calc attenuation between SG and device without atten box
    atten_arr = get_atten_SG2LOout(freq_arr, board=main_dict['board']) + get_atten_insidecryo(freq_arr) + atten_attenbox2cryo 

    # atten_mean = np.mean(atten)
    dev_pwr_max = main_dict['SG382_pwr'] + atten_arr + min_atten_in # max power at the device, if we set attenuation to 0 dB
    dev_pwr_min = main_dict['SG382_pwr'] + atten_arr + max_atten_in # min power at the device, if we set attenuation to -90 dB
    
    atten_in = np.round((main_dict['req_pwr@device'] - main_dict['SG382_pwr'] - atten_arr) / 0.25) * 0.25
    main_dict['atten_in'] = np.array([main_dict['pwr_sweep'] + a for a in atten_in])
    
    dev_pwr = main_dict['SG382_pwr'] + atten_arr[:,None] + main_dict['atten_in']
    S21_gain_dB = get_gain_afterdevice(main_dict, freq_arr)
    RF_pwr = dev_pwr + S21_gain_dB[:,None]

    atten_out = []
    
    for i, freq in enumerate(freq_arr):
        if main_dict['req_pwr@device'][i] > dev_pwr_max[i]:
            print('Power at device requested (%.1f dBm) too high'%(main_dict['req_pwr@device'][i]))
            print('Max power possible at device: %.1f dBm'%dev_pwr_max[i])
            return None
        
        elif main_dict['req_pwr@device'][i] < dev_pwr_min[i]:
            print('Power at device requested (%.1f dBm) too low'%(main_dict['req_pwr@device'][i]))
            print('Min power possible at device: %.1f dBm'%dev_pwr_min[i])    
            return None
            
        if main_dict['pwr_sweep_bool']:
            RF_pwr_min = np.min(RF_pwr[i]) + min_atten_out[i]
            if (RF_pwr_min < main_dict['RFpwr_req@mixer']).any():
                atten_out.append(RF_pwr_min - RF_pwr[i])
                print('Min predicted RF power at mixer input during power sweep: %.1f dBm)'%(RF_pwr_min))
                # print('Attenuator (out) values: %s dB'%(main_dict['atten_out'][i]))
            else:
                atten_out.append(main_dict['RFpwr_req@mixer'] - RF_pwr[i])
                print('Predicted RF power at mixer input: %.1f dBm)'%(main_dict['RFpwr_req@mixer']))
        else:     
            if RF_pwr[i][0] < main_dict['RFpwr_req@mixer']:
                atten_out.append([min_atten_out[i]])
                print('Predicted RF power at mixer input: %.1f dBm)'%(RF_pwr[i][0]))
                # print('Attenuator (out) values: %s dB'%(main_dict['atten_out'][i]))
            else:
                atten_out.append([main_dict['RFpwr_req@mixer'] - RF_pwr[i][0]])
                print('Predicted RF power at mixer input: %.1f dBm)'%(main_dict['RFpwr_req@mixer']))
            
    main_dict['pwr@device'] = dev_pwr # for each reso and each pwr for pwr sweeps
    main_dict['atten_out'] = np.array(atten_out)
    main_dict['RF_pwr'] = RF_pwr + main_dict['atten_out']
    
    args = (main_dict['SG382_pwr'], main_dict['LO_pwr'])
    scan_str = '''\n--------------------------------------------------
    --- Power and attenuation settings --- 
        SG382 signal generator output power: {:.1f} dBm
        LO power (@ mixer input): {:.1f} dBm '''.format(*args)
    # print(scan_str)
    
    for i, freq in enumerate(freq_arr):
        # if main_dict['pwr_sweep_bool']:
        args = (np.array2string(main_dict['pwr@device'][i], precision=1, floatmode='fixed'),
                np.array2string(main_dict['RF_pwr'][i], precision=1, floatmode='fixed'), 
                np.array2string(main_dict['atten_in'][i], precision=1, floatmode='fixed'), 
                np.array2string(main_dict['atten_out'][i], precision=1, floatmode='fixed'))
        # else:
        #     args = (np.array2string(main_dict['pwr@device'][i], precision=1, floatmode='fixed'),
        #             np.array2string(main_dict['RF_pwr'][i], precision=1, floatmode='fixed'), 
        #             np.array2string(main_dict['atten_in'][i], precision=1, floatmode='fixed'), 
        #             np.array2string(main_dict['atten_out'][i], precision=1, floatmode='fixed'))
            
        scan_str += '\n\n    Resonance %.2f MHz: \n'%(freq/1e6)
        scan_str += '''    Power at the device 
            {:s} dBm
        RF power (back to mixer): 
            {:s} dBm
        Variable attenuator in: 
            {:s} dB
        Variable attenuator out: 
            {:s} dB '''.format(*args)
        
    print(scan_str)
    # print('Set Weinschel attenuator to (%d, %d) dBm'%(meas_dict['Weinschel_atten'][0], meas_dict['Weinschel_atten'][1]))
    # wein.setAtten(1, meas_dict['atten_in'])
    # wein.setAtten(2, meas_dict['atten_out'])
    return main_dict


def set_atten(freq, atten_in_req, atten_out_req):
    # freq = main_dict['freq_center']
    atten_margin = 0.3 # dB
    print('Set variable attenuator to reach (%.2f, %.2f) dB attenuation through the attenuator box'%(atten_in_req, atten_out_req))
    
    atten_in = atten_in_req
    atten_out = atten_out_req
    atten_not_reached = True
    i = 0
    
    while atten_not_reached:
        atten_in = np.round(atten_in / 0.25) * 0.25
        atten_out = np.round(atten_out / 0.25) * 0.25
        if atten_in > 0: 
            atten_in = 0
        if atten_out > 0:
            atten_out = 0

        eff_atten_in, eff_atten_out = GetBoxAtten.get_atten_SwitchBox(freq, atten_in, atten_out)
        err_atten_in = np.mean(eff_atten_in) - atten_in_req        
        err_atten_out = np.mean(eff_atten_out) - atten_out_req
        
        if abs(err_atten_in) > atten_margin:
            atten_in = atten_in - err_atten_in
        if abs(err_atten_out) > atten_margin:
            atten_out = atten_out - err_atten_out
        else: 
            atten_not_reached = False
        if i > 20:
            print('Loop does not converge to find the attenuator value, exit!')
            return
        i += 1
    print('Minicircuits attenuators (in, out) set to (%.2f, %.2f) dB, which provides an attenuation through the attenuator box of (%.2f, %.2f) dB'%(atten_in, atten_out, eff_atten_in, eff_atten_out))            
    set_MiniCAtten.setAtten(np.array([atten_in, atten_out]))
    
    
    
def set_SR560(mode):
    SR560_dict = {'Model':'SR560', 'gain_mode':'low_noise'}   

    if mode == 'freq_sweep':
        LP_freq = 3e4 # Hz
        gain = 20
        SR560.set_DC_filters(LP=LP_freq) #comment for two tone
        time.sleep(1)
        SR560.set_gain(gain) 
        SR560_dict['coupling'] = 'DC'
        SR560_dict['low_pass_freq'] = LP_freq
        SR560_dict['low_pass_slope'] = '12 dB/oct'
        SR560_dict['high_pass_freq'] = 0 # Hz, determined by DC mode
        SR560_dict['high_pass_slope'] = 'N/A'
        SR560_dict['gain'] = gain

    elif mode == 'FM':
        LP_freq = 3e4 # Hz
        gain = 20
        SR560.set_DC_filters(LP=LP_freq) #comment for two tone
        time.sleep(1)
        SR560.set_gain(gain) 
        SR560_dict['coupling'] = 'DC'
        SR560_dict['low_pass_freq'] = LP_freq
        SR560_dict['low_pass_slope'] = '12 dB/oct'
        SR560_dict['high_pass_freq'] = 0 # Hz, determined by DC mode
        SR560_dict['high_pass_slope'] = 'N/A'
        SR560_dict['gain'] = gain
        
    elif mode == 'noise' or mode == 'calib':
        LP_freq = 3e5 # Hz
        gain = 100 #change to 100 from 1000 for two tone
        SR560.set_gain(20) 
        SR560.set_AC_LPfilter(LP=LP_freq) 
        time.sleep(1)
        SR560.set_gain(gain) 
        SR560_dict['coupling'] = 'AC'
        SR560_dict['low_pass_freq'] = LP_freq
        SR560_dict['low_pass_slope'] = '6 dB/oct'
        SR560_dict['high_pass_freq'] = 0.03 # Hz, determined by AC mode
        SR560_dict['high_pass_slope'] = '6 dB/oct'
        SR560_dict['gain'] = gain
    

    else: 
        print('Mode not valid (must be either freq_sweep, FM, or noise). Abort')
        return SR560_dict 
    
    args = (SR560_dict['coupling'], SR560_dict['low_pass_freq']/1e3, SR560_dict['low_pass_slope'], SR560_dict['high_pass_freq'], 
    SR560_dict['high_pass_slope'], SR560_dict['gain'], SR560_dict['gain_mode'])
    scan_str = '''\n--------------------------------------------------
    --- SR560 preamps settings --- 
        Coupling: {:s} 
        Low pass frequency: {:.2f} kHz
        Low pass slope: {:s} dB/oct
        High pass frequency: {:.2f} Hz
        High pass slope: {:s}
        Gain: x{:.0f}   
        Gain mode: {:s}
        
    '''.format(*args)
    print(scan_str)
    return SR560_dict


def init_synthesizer_SG382(main_dict, meas_dict):
    rm = pyvisa.ResourceManager()
    SG382 = rm.open_resource("GPIB::9::INSTR") # define as LO
    print('\n --- SG382 signal generator initialization --- ')
    # SG382.write('*RST')
    SG382.write('ENBR 0')
    print('RF output turned OFF')
    time.sleep(0.1)
    SG382.write('MODL 0')
    print('Modulation turned OFF')
    time.sleep(0.1)
    if main_dict['SG382_pwr'] > 16.5: 
        print('Synthe power requested > 16.5 dBm impossible. Synthe power set to 16.5 dBm')
        main_dict['SG382_pwr'] = 16.5
    SG382.write('AMPR %.2f'%main_dict['SG382_pwr'])
    print('SG382 power set to %.2f dBm'%(float(SG382.query('AMPR?'))))
    time.sleep(0.1)
    
    if meas_dict['mode'] == 'freq_sweep':
        # freq_center = (meas_dict['freq_start_array'][0] + meas_dict['freq_stop_array'][0])/2
        # freq_span = meas_dict['freq_stop_array'][0] - meas_dict['freq_start_array'][0]
        SG382.write('FREQ %d'%(meas_dict['freq_center']))
        print('frequency set to %.6f MHz'%(float(SG382.query('FREQ?'))*1e-6))
        time.sleep(0.1)
        SG382.write('TYPE 3') # modulation type (0: AM, 1: FM, 2: PM, 3: Sweep, 4: Pulse, 5: Blank)
        print('Frequency sweep set')
        time.sleep(0.1)
        SG382.write('SFNC 1') # sweep modulation function (0:sine, 1:ramp, 2:triangle, 5:external)
        print('Sweep modulation function: ramp')
        time.sleep(0.1)
        SG382.write('SRAT %.3f'%(meas_dict['rate'])) # Modulation rate, in Hz
        print('Modulation rate = %.2f Hz'%(float(SG382.query('SRAT?'))))
        time.sleep(0.1)        
        SG382.write('SDEV %d'%(meas_dict['span']/2)) # freq_span = 2xSDEV (freq varies from freq - SDEV to freq + SDEV)
        print('Sweep variation set to %.3f MHz'%(float(SG382.query('SDEV?'))*2*1e-6))
        time.sleep(1)
        SG382.write('MODL 1')
        print('Modulation turned ON')
        
    elif meas_dict['mode'] == 'FM':
        SG382.write('FREQ %d'%(meas_dict['freq_center']))
        print('frequency set to %.6f MHz'%(float(SG382.query('FREQ?'))*1e-6))
        time.sleep(0.1)
        SG382.write('TYPE 1') # modulation type (0: AM, 1: FM, 2: PM, 3: Sweep, 4: Pulse, 5: Blank)
        print('Frequency modulation set')
        time.sleep(0.1)
        SG382.write('MFNC 1') # Modulation function (0:sine, 1:ramp, 2:triangle, 3:square, 4:noise,  5:external)
        print('Frequency modulation function: ramp')
        time.sleep(0.1)
        SG382.write('RATE %.3f'%(meas_dict['rate'])) # Modulation rate, in Hz
        print('Modulation rate = %.2f Hz'%(float(SG382.query('RATE?'))))
        time.sleep(0.1)        
        SG382.write('FDEV %d'%(meas_dict['span']/2)) # FM_span = 2xFDEV (freq varies from freq - FDEV to freq + FDEV)
        print('set the FM deviation to %.3f kHz'%(float(SG382.query('FDEV?'))*2*1e-3))
        time.sleep(1)
        SG382.write('MODL 1')
        print('Modulation turned ON')
        
    elif meas_dict['mode'] == 'calib':
        SG382.write('FREQ %d'%(meas_dict['freq_center']))
        print('frequency set to %.6f MHz'%(float(SG382.query('FREQ?'))*1e-6))
        time.sleep(0.1)
        SG382.write('TYPE 1') # modulation type (0: AM, 1: FM, 2: PM, 3: Sweep, 4: Pulse, 5: Blank)
        print('Frequency modulation set')
        time.sleep(0.1)
        SG382.write('MFNC 3') # Modulation function (0:sine, 1:ramp, 2:triangle, 3:square, 4:noise,  5:external)
        print('Frequency modulation function: square')
        time.sleep(0.1)
        SG382.write('RATE %.3f'%(meas_dict['rate'])) # Modulation rate, in Hz
        print('Modulation rate = %.2f Hz'%(float(SG382.query('RATE?'))))
        time.sleep(0.1)        
        SG382.write('FDEV %d'%(meas_dict['span']/2)) # FM_span = 2xFDEV (freq varies from freq - FDEV to freq + FDEV)
        print('set the FM deviation to %.3f kHz'%(float(SG382.query('FDEV?'))*2*1e-3))
        time.sleep(1)
        SG382.write('MODL 1')
        print('Modulation turned ON')
        
    elif meas_dict['mode'] == 'noise':
        SG382.write('FREQ %d'%(meas_dict['freq_center']))
        print('frequency set to %.6f MHz'%(float(SG382.query('FREQ?'))*1e-6))
        time.sleep(0.1)  
        
    else: 
        print('Mode not valid (must be either freq_sweep, FM, or noise). Abort')
        return SG382

    # SG382.write('ENBR 1')
    # print('RF output turned ON')
    return SG382

    
def make_dirs(main_dict):
    meas_dir = main_dict['dev_name'] + '/' + main_dict['time_yyyymmdd'] + '_' + main_dict['time_hhmmss'] + '/'
    main_dict['meas_dir'] = main_dict['dir_path'] + meas_dir
    main_dict['h5_dir'] = main_dict['meas_dir'] + 'h5_data/'
    main_dict['plot_dir'] = main_dict['meas_dir'] + 'Plots/'
    os.makedirs(main_dict['h5_dir'], exist_ok=True) 
    os.makedirs(main_dict['plot_dir'], exist_ok=True) 
    return main_dict
    


def save_data_FM(main_dict, meas_dict, SR560_dict, data_dict, resfit_dict, h5_filepath):
    print('\n --- Saving sweep data as h5 datafile---')

    with h5py.File(h5_filepath, 'a') as hf:  
        gr_source = hf.require_group('source_%s'%(meas_dict['source']))
        gr_FPT = gr_source.require_group('FPT_%.0fmK'%(meas_dict['FPtemp_start']*1e3))
        gr_pwr = gr_FPT.require_group('pwr_%.1fdBm'%(meas_dict['pwr@device']))
        gr_res = gr_pwr.require_group('res_scan')
        if 'data' in gr_res.keys():
            del gr_res['data']
        gr_data = gr_res.create_group('data')
        i_dataset = gr_data.create_dataset('I_data', data=data_dict['I_data'])
        q_dataset = gr_data.create_dataset('Q_data', data=data_dict['Q_data'])
        freq_dataset = gr_data.create_dataset('freq_data', data=data_dict['freq'])
        # Mod_dataset = gr_res.create_dataset('Mod_data', data=mod_data)
        i_dataset.attrs['unit'] = 'V'
        q_dataset.attrs['unit'] = 'V'
        freq_dataset.attrs['unit'] = 'Hz'
        
        for key in main_dict.keys():  
        #04/28/2024 change from '['IQ_mixer', 'dev_name', 'cryo_chan', 'dir_path', 'pwr_sweep', 'LO_pwr', 'board', 
        #'time_yyyymmdd', 'time_hhmmss', 'SG382_pwr']' to main_dict.keys()
            gr_data.attrs[key] = main_dict[key]
        
        gr_SR560 = gr_data.create_group('SR560_dict')
        for key in SR560_dict.keys():
            gr_SR560.attrs[key] = SR560_dict[key]
            
        gr_meas = gr_data.create_group('meas_dict')
        for key in meas_dict.keys():
            gr_meas.attrs[key] = meas_dict[key]
            
        if 'fit' in gr_res.keys():
            del gr_res['fit']
        gr_fit = gr_res.create_group('fit')
        i_fitset = gr_fit.create_dataset('I_data', data=resfit_dict['I_data'])
        q_fitset = gr_fit.create_dataset('Q_data', data=resfit_dict['Q_data'])
        freq_fitset = gr_fit.create_dataset('freq_data', data=resfit_dict['freq_data'])
        # Mod_dataset = gr_res.create_dataset('Mod_data', data=mod_data)
        #for calib test
        # i_fitset = gr_fit.create_dataset('I_data_raw', data=resfit_dict['I_data_raw'])
        # q_fitset = gr_fit.create_dataset('Q_data_raw', data=resfit_dict['Q_data_raw'])
        # freq_fitset = gr_fit.create_dataset('freq_data_raw', data=resfit_dict['freq_data_raw'])
        
        i_fitset.attrs['unit'] = 'V'
        q_fitset.attrs['unit'] = 'V'
        freq_fitset.attrs['unit'] = 'Hz'
        
        for key in ['fres', 'Qi', 'Qc', 'Qr', 'fres_raw', 'Qi_raw', 'Qc_raw', 'Qr_raw']:
            gr_fit.attrs[key] = resfit_dict[key]
            
            
    return 



def save_data_calib(main_dict, meas_dict, SR560_dict, data_dict, h5_filepath):
    print('\n --- Saving sweep data as h5 datafile---')
    # h5_dir = main_dict['h5_dir_sweep']
    # os.makedirs(main_dict['meas_dir'] + h5_dir, exist_ok=True) 
    if meas_dict['mode'] == 'calib':
        with h5py.File(h5_filepath, 'a') as hf:  
            gr_source = hf.require_group('source_%s'%(meas_dict['source']))
            gr_FPT = gr_source.require_group('FPT_%.0fmK'%(meas_dict['FPtemp_start']*1e3))
            gr_pwr = gr_FPT.require_group('pwr_%.1fdBm'%(meas_dict['pwr@device']))
            gr_res = gr_pwr.require_group('res_scan')
            if 'calib' in gr_res.keys():
                del gr_res['calib']
            gr_calib = gr_res.create_group('calib')
            i_dataset = gr_calib.create_dataset('I_data', data=data_dict['I_data'])
            q_dataset = gr_calib.create_dataset('Q_data', data=data_dict['Q_data'])
            Mod_dataset = gr_calib.create_dataset('Mod_data', data=data_dict['Mod_data'])
            angle_dataset = gr_calib.create_dataset('angle', data=data_dict['angle'])
            dIQ_vs_df_dataset = gr_calib.create_dataset('dIQ_vs_df', data=data_dict['dIQ_vs_df'])

            i_dataset.attrs['unit'] = 'V'
            q_dataset.attrs['unit'] = 'V'
            Mod_dataset.attrs['unit'] = 'V'
            angle_dataset.attrs['unit'] = 'rad'
            dIQ_vs_df_dataset.attrs['unit'] = 'V/Hz'
                       
            for key in ['IQ_mixer', 'dev_name', 'cryo_chan', 'dir_path', 
                        'pwr_sweep', 'LO_pwr', 'board', 'time_yyyymmdd', 
                        'time_hhmmss', 'SG382_pwr']:#main_dict.keys():
                gr_calib.attrs[key] = main_dict[key]
                
                
            gr_SR560 = gr_calib.create_group('SR560_dict')
            for key in SR560_dict.keys():
                gr_SR560.attrs[key] = SR560_dict[key]
                
            gr_meas = gr_calib.create_group('meas_dict')
            for key in meas_dict.keys():
                gr_meas.attrs[key] = meas_dict[key]
    return


def save_data_noise(main_dict, meas_dict, SR560_dict, data_dict, h5_filepath, i):
    if meas_dict['mode'] == 'noise':
        with h5py.File(h5_filepath, 'a') as hf:  
            gr_source = hf.require_group('source_%s'%(meas_dict['source']))
            gr_FPT = gr_source.require_group('FPT_%.0fmK'%(meas_dict['FPtemp_start']*1e3))
            gr_pwr = gr_FPT.require_group('pwr_%.1fdBm'%(meas_dict['pwr@device']))
            gr_res = gr_pwr.require_group('res_scan')
            if 'noise' in gr_res.keys():
                gr_noise = gr_res.require_group('noise')
            else:
                gr_noise = gr_res.create_group('noise')
                gr_meas = gr_noise.create_group('meas_dict')
                for key in meas_dict.keys():
                    gr_meas.attrs[key] = meas_dict[key]
                    
            i_onres_dataset = gr_noise.create_dataset('I_onres'+str(i), data=data_dict['I_onres'])
            q_onres_dataset = gr_noise.create_dataset('Q_onres'+str(i), data=data_dict['Q_onres'])
            i_offres_dataset = gr_noise.create_dataset('I_offres'+str(i), data=data_dict['I_offres'])
            q_offres_dataset = gr_noise.create_dataset('Q_offres'+str(i), data=data_dict['Q_offres'])
            
            i_onres_dataset.attrs['unit'] = 'V'
            q_onres_dataset.attrs['unit'] = 'V'
            i_offres_dataset.attrs['unit'] = 'V'
            q_offres_dataset.attrs['unit'] = 'V'
            
            # if 'noise_raw' in gr_res.keys():
            #     gr_noise = gr_res.require_group('noise_raw')
            # else:
            #     gr_noise = gr_res.create_group('noise_raw')
            
            # # gr_noise = gr_res.create_group('noise_raw')
            # i_onres_dataset = gr_noise.create_dataset('I_onres'+str(i), data=data_dict['I_onres_raw'])
            # q_onres_dataset = gr_noise.create_dataset('Q_onres'+str(i), data=data_dict['Q_onres_raw'])
            # i_offres_dataset = gr_noise.create_dataset('I_offres'+str(i), data=data_dict['I_offres_raw'])
            # q_offres_dataset = gr_noise.create_dataset('Q_offres'+str(i), data=data_dict['Q_offres_raw'])
            
            # i_onres_dataset.attrs['unit'] = 'V'
            # q_onres_dataset.attrs['unit'] = 'V'
            # i_offres_dataset.attrs['unit'] = 'V'
            # q_offres_dataset.attrs['unit'] = 'V'
            
    return


def save_data_Sdff(main_dict, meas_dict, SR560_dict, Sdff_dict, h5_filepath, N_noise):
    # if h5_filepath == '':
    #     fres_str = ('%.2fMHz'%(meas_dict['freq_center']/1e6)).replace('.', 'p')
    #     filename_h5 = main_dict['time_hhmmss'] + '_IQmixer_res_%s.h5'%fres_str
    #     h5_filepath = main_dict['h5_dir'] + filename_h5

    with h5py.File(h5_filepath, 'a') as hf:  
        gr_source = hf.require_group('source_%s'%(meas_dict['source']))
        gr_FPT = gr_source.require_group('FPT_%.0fmK'%(meas_dict['FPtemp_start']*1e3))
        gr_pwr = gr_FPT.require_group('pwr_%.1fdBm'%(meas_dict['pwr@device']))
        gr_res = gr_pwr.require_group('res_scan')
        if 'Sdff' in gr_res.keys():
            del gr_res['Sdff']
        gr_Sdff = gr_res.create_group('Sdff')
        f_dataset = gr_Sdff.create_dataset('f', data=Sdff_dict['f'])
        PSD_onres_freq_dataset = gr_Sdff.create_dataset('PSD_onres_freq_avg', data=Sdff_dict['PSD_onres_freq_avg'])
        PSD_onres_diss_dataset = gr_Sdff.create_dataset('PSD_onres_diss_avg', data=Sdff_dict['PSD_onres_diss_avg'])
        PSD_offres_freq_dataset = gr_Sdff.create_dataset('PSD_offres_freq_avg', data=Sdff_dict['PSD_offres_freq_avg'])
        PSD_offres_diss_dataset = gr_Sdff.create_dataset('PSD_offres_diss_avg', data=Sdff_dict['PSD_offres_diss_avg'])
        # for i in range(N_noise):
        #     PSD_onres_freq_dataset = gr_Sdff.create_dataset('PSD_onres_freq'+str(i), data=Sdff_dict['PSD_onres_freq'+str(i)])
        #     PSD_onres_diss_dataset = gr_Sdff.create_dataset('PSD_onres_diss'+str(i), data=Sdff_dict['PSD_onres_diss'+str(i)])
        #     PSD_offres_freq_dataset = gr_Sdff.create_dataset('PSD_offres_freq'+str(i), data=Sdff_dict['PSD_offres_freq'+str(i)])
        #     PSD_offres_diss_dataset = gr_Sdff.create_dataset('PSD_offres_diss'+str(i), data=Sdff_dict['PSD_offres_diss'+str(i)])
        
        f_dataset.attrs['unit'] = 'Hz'
        PSD_onres_freq_dataset.attrs['unit'] = 'Hz/Hz'
        PSD_onres_diss_dataset.attrs['unit'] = 'Hz/Hz'
        PSD_offres_freq_dataset.attrs['unit'] = 'Hz/Hz'
        PSD_offres_diss_dataset.attrs['unit'] = 'Hz/Hz'
        
        gr_meas = gr_Sdff.create_group('meas_dict')
        for key in meas_dict.keys():
            gr_meas.attrs[key] = meas_dict[key]
            
        gr_SR560 = gr_Sdff.create_group('SR560_dict')
        for key in SR560_dict.keys():
            gr_SR560.attrs[key] = SR560_dict[key]
        
    return 

def ini_plot():
    fig, axs = plt.subplots(1, 2, figsize=(17, 8), dpi=100)
    plt.subplots_adjust(bottom=0.12, top=0.9, right=0.96, left=0.10, wspace=0.28)
    title_str = fig.text(0.4, 0.94, '', fontsize=16)
    line1, = axs[0].plot([], [], '-', markersize=3, color='C0')
    axs[0].set_xlabel('I [V]', fontsize=16)
    axs[0].set_ylabel('Q [V]', fontsize=16)
    axs[0].tick_params(color='k', labelsize =14)
    axs[0].tick_params(color='k', labelsize =14)
    axs[0].axis('equal')
    axs[0].grid()
    line2, = axs[1].plot([], [], '-', linewidth=2, color='C0')
    axs[1].set_xlabel('Frequency [MHz]', fontsize=16)
    axs[1].set_ylabel(r'$|S_{21}|^2$ [dBV]', fontsize=16)
    axs[1].tick_params(color='k', labelsize =14)
    axs[1].tick_params(color='k', labelsize =14)
    axs[1].grid()
    plt.pause(1)
    plt.show()
    return fig, axs, [line1, line2], title_str


def update_plot(DAQ_data_tmp, LO_freq_arr, axs, lines, title_str):
    I, Q = DAQ_data_tmp[:,0], DAQ_data_tmp[:,1]
    S21_mag_dB = 10*np.log10(I**2 + Q**2)
    freq_mean = np.mean(LO_freq_arr)
    title_str.set_text('Frequency sweep center: %.3f MHz'%(freq_mean/1e6))
    lines[0].set_data(I, Q)
    axs[0].relim()
    axs[0].autoscale_view()     
    lines[1].set_data(LO_freq_arr/1e6, S21_mag_dB)
    axs[1].relim()
    axs[1].autoscale_view()  
    plt.pause(0.05)
    plt.draw()
    plt.show()
    


def check_FM_params(meas_dict):#, i_res):
    # FM_rate = meas_dict['FM_rate']
    FM_rate = meas_dict['rate']
    FM_center_freq = meas_dict['freq_center']#[i_res]
    # FM_span = meas_dict['FM_span']
    FM_span = meas_dict['span']
    freq_max = 2e9 # maximum allowed frequency (2 GHz), in Hz
    rate_range = [1e-6, 50e3] # allowed rate range, in Hz
    max_span_list = [1e6, 2e6, 4e6, 8e6, 16e6] # maximum allowed FM span for different frequency ranges, in Hz
    freq4max_span_list = np.array([126e6, 253e6, 506e6, 1012e6, 2000e6]) # freq range for maximum allowed span, in Hz
    if FM_rate < rate_range[0]:
        print('requested rate too low, rate set to %.2e Hz'%rate_range[0])
        FM_rate = rate_range[0]
    elif FM_rate > rate_range[1]:
        print('requested rate too high, rate set to %d kHz'%(rate_range[1]/1e3))
        FM_rate = rate_range[1]
    
    span_i = np.argmin(np.abs(FM_center_freq - freq4max_span_list))
    if FM_center_freq - freq4max_span_list[span_i] > 0:
        span_i += 1
        
    if FM_span > max_span_list[span_i]:
        print('requested span too large, span set to %d MHz'%(max_span_list[span_i]/1e6))
        FM_span = max_span_list[span_i]

    if FM_center_freq > freq_max:
        print('requested frequency too high, frequency set to %d MHz'%(freq_max/1e6))
        FM_center_freq = freq_max

    meas_dict['freq_center'] = FM_center_freq
    meas_dict['span'] = FM_span
    meas_dict['rate'] = FM_rate
    # meas_dict['FM_span'] = FM_span
    # meas_dict['FM_rate'] = FM_rate
    return meas_dict
    

def take_freq_sweep(main_dict, meas_dict, SG382, task):
    SG382.write('ENBR 1')
    print('SG382 RF output turned ON')
    time.sleep(2)
    
    data_dict = {'freq':[], 'I_data':[], 'Q_data':[], 'Mod_data':[]}
    print('\nFrequency sweep %.2f MHz - %.2f MHz: '%((meas_dict['freq_start']/1e6, meas_dict['freq_stop']/1e6)))
    for i_band in range(meas_dict['N_bands']):
        freq_start = meas_dict['freq_start_array'][i_band]
        freq_stop = meas_dict['freq_stop_array'][i_band]
        freq_center = (freq_start + freq_stop)/2
        freq_span = freq_stop - freq_start
        print('\nScanning band %d/%d'%(i_band+1, meas_dict['N_bands']))
        print('Frequency sweep: %.2f - %.2f MHz'%(freq_start/1e6, freq_stop/1e6))
        
        SG382.write('FREQ %d'%(freq_center))
        print('frequency set to %.6f MHz'%(float(SG382.query('FREQ?'))*1e-6))
        SG382.write('SDEV %d'%(freq_span/2)) # The signal sweeps from freq - sdev to freq + sdev
        print('set the sweep range to %.3f MHz'%(float(SG382.query('SDEV?'))*2*1e-6))
        time.sleep(0.1)
        daq_data = daq_get_data(main_dict, meas_dict, task)
        i_start = np.argmin(daq_data[2,0:meas_dict['Npts_period']]) 
        i_stop = np.argmax(daq_data[2,i_start:]) + i_start
        Npts_period_meas = i_stop - i_start
        freq = np.linspace(freq_start, freq_stop, Npts_period_meas)

        I_data = daq_data[0,i_start:i_stop]
        Q_data = daq_data[1,i_start:i_stop]
        Mod_data = daq_data[2,i_start:i_stop]
        
        data_dict['freq'] = np.concatenate([data_dict['freq'], freq])
        data_dict['I_data'] = np.concatenate([data_dict['I_data'], I_data])
        data_dict['Q_data'] = np.concatenate([data_dict['Q_data'], Q_data])
        data_dict['Mod_data'] = np.concatenate([data_dict['Mod_data'], Mod_data])
    
    SG382.write('ENBR 0')
    print('SG382 RF output turned OFF')
    return data_dict
    
    
def update_meas_dict(main_dict, meas_dict, i_res=0, i_pwr=0):

    if meas_dict['mode'] == 'freq_sweep':
        rate = main_dict['sweep_rate']
        span = main_dict['sweep_span']
    elif meas_dict['mode'] == 'FM':
        rate = main_dict['FM_rate']
        span = main_dict['FM_span']
    elif meas_dict['mode'] == 'calib':
        rate = main_dict['calib_rate']
        span = main_dict['calib_span']
        
    if hasattr(rate, "__len__"): # check if rate is an array or a float
        meas_dict['rate'] = rate[i_res]
    else:
        meas_dict['rate'] = rate
        
    if hasattr(span, "__len__"): # check if span is an array or a float
        meas_dict['span'] = span[i_res]
    else:
        meas_dict['span'] = span
    
    if hasattr(main_dict['freq_center'], "__len__"): # check if main_dict['freq_center'] is an array or a float
        meas_dict['freq_center'] = main_dict['freq_center'][i_res]
        meas_dict['freq_start'] = main_dict['freq_center'][i_res] - meas_dict['span'] / 2
        meas_dict['freq_stop'] = main_dict['freq_center'][i_res] + meas_dict['span'] / 2
    else:
        meas_dict['freq_center'] = main_dict['freq_center']
        meas_dict['freq_start'] = main_dict['freq_center'] - meas_dict['span'] / 2
        meas_dict['freq_stop'] = main_dict['freq_center'] + meas_dict['span'] / 2        

    # if main_dict['pwr_sweep_bool']:
    meas_dict['atten_in'] = main_dict['atten_in'][i_res][i_pwr]
    meas_dict['atten_out'] = main_dict['atten_out'][i_res][i_pwr]
    # else:
    #     meas_dict['atten_in'] = main_dict['atten_in'][i_res]
    #     meas_dict['atten_out'] = main_dict['atten_out'][i_res]   
    
    return meas_dict


def set_meas_dict(main_dict, mode):
    meas_dict = {'mode':mode}
    update_meas_dict(main_dict, meas_dict, i_res=0, i_pwr=0)
    return meas_dict
        

def find_freq_sweep_reso(main_dict, data_dict):
    for i_res in range(main_dict['Nres']):
        S21_dB = 10*np.log10(data_dict[i_res]['I_data']**2 + data_dict[i_res]['Q_data']**2)
        peaks, _ = sig.find_peaks(-1*S21_dB, prominence=2, distance=1e3) # distance is the min distance between peaks
        fres = data_dict[i_res]['freq'][peaks]
        i_fres_target = np.argmin(np.abs(fres - main_dict['freq_center'][i_res]))
        data_dict[i_res]['fres'] = data_dict[i_res]['freq'][peaks]
        data_dict[i_res]['fres_target'] = data_dict[i_res]['fres'][i_fres_target]
    return data_dict

        
def freq_sweep_main(main_dict, meas_dict, plot=True):

    mode = 'freq_sweep'
    meas_dict = set_meas_dict(main_dict, mode) 
    SG382 = init_synthesizer_SG382(main_dict, meas_dict) # Initialize the signal generator, but set RF off
    task, meas_dict = init_daq(meas_dict)
    SR560_dict = set_SR560(mode)
    # main_dict = define_atten_and_pwr(main_dict)
    time.sleep(2)
    data_dict = []
    for i_res in range(main_dict['Nres']):
        meas_dict = update_meas_dict(main_dict, meas_dict, i_res=i_res, i_pwr=0)
        meas_dict = define_freq_sweep_bands(meas_dict)
        set_atten(meas_dict['freq_center'], meas_dict['atten_in'], meas_dict['atten_out'])
        data_dict.append(take_freq_sweep(main_dict, meas_dict, SG382, task))
        
        S21_dB = 10*np.log10(data_dict[i_res]['I_data']**2 + data_dict[i_res]['Q_data']**2)
        peaks, _ = sig.find_peaks(-1*S21_dB, prominence=2, distance=1e3) # distance is the min distance between peaks
        fres = data_dict[i_res]['freq'][peaks]
        i_fres_target = np.argmin(np.abs(fres - main_dict['freq_center'][i_res]))
        data_dict[i_res]['fres'] = data_dict[i_res]['freq'][peaks]
        data_dict[i_res]['fres_target'] = data_dict[i_res]['fres'][i_fres_target]
        
    SG382.write('ENBR 0') # Set SG382 output Off
    task.close()  
    time.sleep(2)
    # data_dict = find_freq_sweep_reso(main_dict, data_dict)
    main_dict['freq_center'] = np.array([a['fres_target'] for a in data_dict])
    main_dict['freq_center_str'] = [('%.2f'%(a['fres_target']/1e6)).replace('.','p') for a in data_dict]
    data_dict_calib = data_dict

    if plot == True:
        figs = []
        for i_res in range(main_dict['Nres']):
            freq = data_dict[i_res]['freq']
            freq_onres = data_dict[i_res]['fres']
            freq_onres_target = data_dict[i_res]['fres_target']
            freq_offres = freq_onres_target + main_dict['df_onoff_res']
            
            i_onres = [np.argmin(abs(a - freq)) for a in freq_onres]
            i_onres_target = np.argmin(abs(freq_onres_target - freq))
            i_offres = np.argmin(abs(freq_offres - freq))
            
            S21_dB = 10*np.log10(data_dict[i_res]['I_data']**2 + data_dict[i_res]['Q_data']**2)
            # peaks, _ = sig.find_peaks(-1*S21_dB, prominence=5)
            fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
            plt.subplots_adjust(bottom=0.12, top=0.9, right=0.96, left=0.10, wspace=0.28)
            ax.set_xlabel('Frequency [MHz]', fontsize=16)
            ax.set_ylabel(r'$|S_{21}|$ [dBV]', fontsize=16)
            ax.tick_params(color='k', labelsize =14)
            ax.tick_params(color='k', labelsize =14)
            ax.plot(freq/1e6, S21_dB, 'C0', linewidth=2, label='raw')
            freq_onres_str = ''
            for peak_freq in freq_onres/1e6:
                freq_onres_str += '%.3f MHz\n'%peak_freq
            ax.plot(freq[i_onres]/1e6, S21_dB[i_onres], 'ok', linewidth=2, label=freq_onres_str)
            ax.plot(freq[i_onres_target]/1e6, S21_dB[i_onres_target], '*C3', markersize=12, label = 'On res')
            ax.plot(freq[i_offres]/1e6, S21_dB[i_offres], '*C2', markersize = 12, label = 'Off res')
            
            data_dict_calib[i_res]['I_data'], data_dict_calib[i_res]['Q_data'] = main_calib_IQ_DCdata_d(data_dict[i_res]['I_data'], data_dict[i_res]['Q_data'], data_dict[i_res]['freq'], IQ_calibfile_path=IQ_calibfileDC_path)
            S21_dB = 10*np.log10(data_dict_calib[i_res]['I_data']**2 + data_dict_calib[i_res]['Q_data']**2)
            ax.plot(freq/1e6, S21_dB, 'r', linestyle=':', linewidth=2, label='add IQ Mixer calib')
            figtitle = '%s:  %dmK  %ddBm  on res:%.2fMHz  off res:%.2fMHz'%(main_dict['dev_name'], int(main_dict['FPtemp_ini']*1000), int(main_dict['req_pwr@device'][0]), freq[i_onres_target]/1e6, freq[i_offres]/1e6)
            fig.text(0.5, 0.95, figtitle, fontsize=14, horizontalalignment='center', verticalalignment='top')
            ax.legend(fontsize=14)
            ax.grid()
            plt.show(block=False)
            figs.append(fig)
            
        for i,fig in enumerate(figs):
            sweep_dir = main_dict['plot_dir'] + 'freq_sweep/'
            os.makedirs(sweep_dir, exist_ok=True)
            filename = sweep_dir + 'res_%sMHz'%main_dict['freq_center_str'][i]
            print('Saving %s MHz sweep'%main_dict['freq_center_str'][i])
            fig.savefig(filename + '.pdf', dpi=100)#, bbox_inches = 'tight')
            # if svg: fig.savefig(filename + '.svg', dpi=mydpi)#, bbox_inches = 'tight')
            # fig.savefig(filename + '.png', dpi=mydpi)#, bbox_inches = 'tight')
    return data_dict, meas_dict
 

def fit_res(data_dict, mode='lmfit'):
    freq = data_dict['freq']
    I = data_dict['I_data']
    Q = data_dict['Q_data']
    N = len(freq)
    df = freq[1] - freq[0]
    dataDict = {'I':I, 'Q':Q, 'temp':0.240, 
                'pwr':0, 'freq':freq, 'name':'Be000000bl'}
    resObj = scr.makeResFromData(dataDict)                                
    resObj.load_params(scr.cmplxIQ_params, fit_quadratic_phase = True, hardware = 'VNA') # hardware = VNA or mixer
    dfres = resObj.params['f0'].value * (1/resObj.params['qi'].value + 1/resObj.params['qc'].value)
    fres_arg = np.argmin(abs(freq - resObj.params['f0'].value))    
    df_arg = min([max([int(20*dfres/df), int(200e3/df)]), fres_arg, N-fres_arg])
    f_start_arg = fres_arg - df_arg
    f_stop_arg = fres_arg + df_arg
    for key in ['I', 'Q', 'freq']:
        dataDict[key] = dataDict[key][f_start_arg:f_stop_arg]
    resObj = scr.makeResFromData(dataDict)                                
    resObj.load_params(scr.cmplxIQ_params, fit_quadratic_phase = True, hardware = 'VNA')
    resObj.do_lmfit(scr.cmplxIQ_fit)
    
    fit_params = resObj.lmfit_result['default']['result']
    resfit_dict = {}
    resfit_dict['I_data_raw'] = resObj.I
    resfit_dict['Q_data_raw'] = resObj.Q
    resfit_dict['freq_data_raw'] = resObj.freq
    resfit_dict['fres_raw'] = fit_params.params['f0'].value + fit_params.params['df'].value
    resfit_dict['Qi_raw'] = fit_params.params['qi'].value
    resfit_dict['Qc_raw'] = fit_params.params['qc'].value
    resfit_dict['Qr_raw'] = 1/(1/fit_params.params['qi'].value + 1/fit_params.params['qc'].value)
    resfit_dict['resObj_raw'] = resObj
    resfit_dict['fit_params_raw'] = fit_params
    
    freq = data_dict['freq']
    I = data_dict['I_data_calib']
    Q = data_dict['Q_data_calib']
    N = len(freq)
    df = freq[1] - freq[0]
    dataDict = {'I':I, 'Q':Q, 'temp':0.240, 
                'pwr':0, 'freq':freq, 'name':'Be000000bl'}
    resObj = scr.makeResFromData(dataDict)                                
    resObj.load_params(scr.cmplxIQ_params, fit_quadratic_phase = True, hardware = 'VNA') # hardware = VNA or mixer
    dfres = resObj.params['f0'].value * (1/resObj.params['qi'].value + 1/resObj.params['qc'].value)
    fres_arg = np.argmin(abs(freq - resObj.params['f0'].value))    
    df_arg = min([max([int(20*dfres/df), int(200e3/df)]), fres_arg, N-fres_arg])
    f_start_arg = fres_arg - df_arg
    f_stop_arg = fres_arg + df_arg
    for key in ['I', 'Q', 'freq']:
        dataDict[key] = dataDict[key][f_start_arg:f_stop_arg]
    resObj = scr.makeResFromData(dataDict)                                
    resObj.load_params(scr.cmplxIQ_params, fit_quadratic_phase = True, hardware = 'VNA')
    resObj.do_lmfit(scr.cmplxIQ_fit)
    
    fit_params = resObj.lmfit_result['default']['result']
    resfit_dict['I_data'] = resObj.I
    resfit_dict['Q_data'] = resObj.Q
    resfit_dict['freq_data'] = resObj.freq
    resfit_dict['fres'] = fit_params.params['f0'].value + fit_params.params['df'].value
    resfit_dict['Qi'] = fit_params.params['qi'].value
    resfit_dict['Qc'] = fit_params.params['qc'].value
    resfit_dict['Qr'] = 1/(1/fit_params.params['qi'].value + 1/fit_params.params['qc'].value)
    resfit_dict['resObj'] = resObj
    resfit_dict['fit_params'] = fit_params
    
    return resfit_dict

def calc_res_fit(freq, resObj, fit_params):
    '''
    Get the cable delay (total_gain) by using the formula in SCRAPS library
    '''
    print('\nCalculating the cable delay...')
    qi = fit_params.params['qi'].value
    qc = fit_params.params['qc'].value
    qr = 1/(1/qi+1/qc)
    df = fit_params.params['df'].value
    f0 = fit_params.params['f0'].value
    gain0 = fit_params.params['gain0'].value
    gain1 = fit_params.params['gain1'].value
    gain2 = fit_params.params['gain2'].value
    pgain0 = fit_params.params['pgain0'].value
    pgain1 = fit_params.params['pgain1'].value
    pgain2 = fit_params.params['pgain2'].value
    fm = resObj.freq[int(np.round((len(resObj.freq)-1)/2))]
    ffm = (freq-fm)/fm
    gain = gain0 + gain1*ffm+ 0.5*gain2*ffm**2
    pgain = np.exp(1j*(pgain0 + pgain1*ffm + 0.5*pgain2*ffm**2))
    fs = f0+df
    ff = (freq-fs)/fs
    S21_norm_fit = (1./qi+1j*2.0*(ff+df/fs))/(1./qr+1j*2.0*ff)
    total_gain = gain*pgain
    return total_gain, S21_norm_fit 

def FM_main(main_dict, meas_dict):#, i_res):

    data_dict = {}
    meas_dict = check_FM_params(meas_dict)#, i_res)
    # main_dict = init_measurement(main_dict)
    # meas_dict = set_atten_and_pwr(main_dict, meas_dict)
    SR560_dict = set_SR560(meas_dict['mode'])
    time.sleep(2)
    SG382 = init_synthesizer_SG382(main_dict, meas_dict)
    SG382.write('ENBR 1')
    task, meas_dict = init_daq(meas_dict)
    time.sleep(0.5)
    
    print('Frequency Modulation measurement: ')
    print('Center freq: %.3f MHz, FM deviation = %.2f kHz'%(meas_dict['freq_center']/1e6, meas_dict['span']/1e3))
    
    daq_data = daq_get_data(main_dict, meas_dict, task)
    # daq_data[1] = daq_data[1]*-1 # we take the inverse of Q data, because the resonance fit fails if we don't
    i_start = np.argmin(daq_data[2,0:meas_dict['Npts_period']]) 
    i_stop = np.argmax(daq_data[2,i_start:]) + i_start
    Npts_period_meas = i_stop - i_start
    freq_start = meas_dict['freq_center'] - meas_dict['span']/2
    freq_stop = meas_dict['freq_center'] + meas_dict['span']/2
    
    data_dict['freq'] = np.linspace(freq_start, freq_stop, Npts_period_meas)
    data_dict['I_data'] = daq_data[0,i_start:i_stop]
    data_dict['Q_data'] = daq_data[1,i_start:i_stop]
    data_dict['Mod_data'] = daq_data[2,i_start:i_stop]
    
    data_dict['I_data_calib'], data_dict['Q_data_calib'] = main_calib_IQ_DCdata_d(data_dict['I_data'], data_dict['Q_data'], data_dict['freq'], IQ_calibfile_path=IQ_calibfileDC_path)
    
    SG382.write('ENBR 0')
    task.close()

    return main_dict, meas_dict, SR560_dict, data_dict
    

def plot_fit_reso(data_dict, resfit_dict, main_dict, meas_dict):
    fig, axs = plt.subplots(1, 2, figsize=(17, 10), dpi=100)
    plt.subplots_adjust(bottom=0.12, top=0.9, right=0.96, left=0.10, wspace=0.28)
    # title_str = fig.text(0.4, 0.94, '', fontsize=16)
    axs[0].set_xlabel('I [V]', fontsize=16)
    axs[0].set_ylabel('Q [V]', fontsize=16)
    axs[0].tick_params(color='k', labelsize =14)
    axs[0].tick_params(color='k', labelsize =14)
    axs[0].axis('equal')
    axs[0].grid()
    axs[1].set_xlabel('Frequency [MHz]', fontsize=16)
    axs[1].set_ylabel(r'$|S_{21}|$ [dBV]', fontsize=16)
    axs[1].tick_params(color='k', labelsize =14)
    axs[1].tick_params(color='k', labelsize =14)
    axs[1].grid()
    
    # resfreq_data = resfit_dict['freq_data']
    # freq_interp_sweep = np.linspace(resfreq_data[0], resfreq_data[-1], len(resfreq_data)) 
    # baseline_sweep_fit, S21_sweep_fit_norm = calc_res_fit(freq_interp_sweep, resfit_dict['resObj'], resfit_dict['fit_params'])
    # baseline_sweep_data, _ = calc_res_fit(resfit_dict['freq_data'], resfit_dict['resObj'], resfit_dict['fit_params'])
    # fres = resfit_dict['fres']
    # i_fres = np.argmin(np.abs(fres - freq_interp_sweep))
    # S21_sweep_data_norm = (resfit_dict['I_data'][0] + 1j*resfit_dict['Q_data'][0]) / baseline_sweep_data
    # S21_data_norm_mag_dB = 20 * np.log10(np.abs(S21_sweep_data_norm))
    # S21_fit_norm_mag_dB = 20 * np.log10(np.abs(baseline_sweep_fit))
    # axs[0].plot(np.real(S21_sweep_data_norm), np.imag(S21_sweep_data_norm), '.C0', markersize=3, label='data add IQ Mixer calib')
    # axs[0].plot(np.real(S21_sweep_fit_norm), np.imag(S21_sweep_fit_norm), 'k', linewidth=2, label='fit add IQ Mixer calib')
    # axs[0].plot(np.real(S21_sweep_fit_norm[i_fres]), np.imag(S21_sweep_fit_norm[i_fres]), '*C1', markersize=10, label='fres add IQ Mixer calib')
    # axs[1].plot(resfit_dict['freq_data']/1e6, S21_data_norm_mag_dB, '.C0', markersize=3, label='data add IQ Mixer calib')
    # axs[1].plot(freq_interp_sweep/1e6, S21_fit_norm_mag_dB, 'k', linewidth=2, label='fit add IQ Mixer calib')
    # axs[1].plot(fres/1e6, S21_fit_norm_mag_dB[i_fres], '*C1', markersize=10, label='fres add IQ Mixer calib')
    
    # resfreq_data = resfit_dict['freq_data_raw']
    # freq_interp_sweep = np.linspace(resfreq_data[0], resfreq_data[-1], len(resfreq_data)) 
    # baseline_sweep_fit, S21_sweep_fit_norm = calc_res_fit(freq_interp_sweep, resfit_dict['resObj_raw'], resfit_dict['fit_params_raw'])
    # baseline_sweep_data, _ = calc_res_fit(resfit_dict['freq_data_raw'], resfit_dict['resObj_raw'], resfit_dict['fit_params_raw'])
    # fres = resfit_dict['fres_raw']
    # i_fres = np.argmin(np.abs(fres - freq_interp_sweep))
    # S21_sweep_data_norm = (resfit_dict['I_data_raw'][0] + 1j*resfit_dict['Q_data_raw'][0]) / baseline_sweep_data
    # S21_data_norm_mag_dB = 20 * np.log10(np.abs(S21_sweep_data_norm))
    # S21_fit_norm_mag_dB = 20 * np.log10(np.abs(baseline_sweep_fit))
    # axs[0].plot(np.real(S21_sweep_data_norm), np.imag(S21_sweep_data_norm), '.g', markersize=3, label='data_raw')
    # axs[0].plot(np.real(S21_sweep_fit_norm), np.imag(S21_sweep_fit_norm), 'r', linewidth=2, label='fit_raw')
    # axs[0].plot(np.real(S21_sweep_fit_norm[i_fres]), np.imag(S21_sweep_fit_norm[i_fres]), '*b', markersize=10, label='fres_raw')
    # axs[1].plot(resfit_dict['freq_data_raw']/1e6, S21_data_norm_mag_dB, '.g', markersize=3, label='data_raw')
    # axs[1].plot(freq_interp_sweep/1e6, S21_fit_norm_mag_dB, 'r', linewidth=2, label='fit_raw')
    # axs[1].plot(fres/1e6, S21_fit_norm_mag_dB[i_fres], '*b', markersize=10, label='fres_raw')
    
    # baseline_fit, S21_norm_fit = calc_res_fit(resfit_dict['freq_data'], resfit_dict['resObj'], resfit_dict['fit_params'])
    # baseline_data, S21_norm_data = calc_res_fit(resfit_dict['freq_data'], resfit_dict['resObj'], resfit_dict['fit_params'])
    baseline_data, S21_norm_fit = calc_res_fit(resfit_dict['freq_data'], resfit_dict['resObj'], resfit_dict['fit_params'])
    fres = resfit_dict['fres']
    i_fres = np.argmin(np.abs(fres - resfit_dict['freq_data']))
    S21_data_norm = (resfit_dict['I_data'] + 1j*resfit_dict['Q_data']) / baseline_data
    S21_data_norm_mag_dB = 20 * np.log10(np.abs(S21_data_norm))
    S21_fit_norm_mag_dB = 20 * np.log10(np.abs(S21_norm_fit))
    axs[0].plot(np.real(S21_data_norm), np.imag(S21_data_norm), '.C0', markersize=3, label='data add IQ Mixer calib')
    axs[0].plot(np.real(S21_norm_fit), np.imag(S21_norm_fit), 'k', linewidth=2, label='fit add IQ Mixer calib')
    axs[0].plot(np.real(S21_norm_fit[i_fres]), np.imag(S21_norm_fit[i_fres]), '*C1', markersize=10, label='fres add IQ Mixer calib')
    axs[1].plot(resfit_dict['freq_data']/1e6, S21_data_norm_mag_dB, '.C0', markersize=3, label='data add IQ Mixer calib')
    axs[1].plot(resfit_dict['freq_data']/1e6, S21_fit_norm_mag_dB, 'k', linewidth=2, label='fit add IQ Mixer calib')
    axs[1].plot(fres/1e6, S21_fit_norm_mag_dB[i_fres], '*C1', markersize=10, label='fres add IQ Mixer calib')
    
    # S21_data_mag_dB = 20 * np.log10(np.sqrt(np.square(data_dict['I_data_calib'])+np.square(data_dict['Q_data_calib'])))
    # axs[0].plot(data_dict['I_data_calib'], data_dict['Q_data_calib'], '.C0', markersize=3, label='data add IQ Mixer calib')
    # axs[0].plot(np.real(S21_data_norm), np.imag(S21_data_norm), 'k', linewidth=2, label='fit add IQ Mixer calib')
    # axs[0].plot(np.real(S21_data_norm[i_fres]), np.imag(S21_data_norm[i_fres]), '*C1', markersize=10, label='fres add IQ Mixer calib')
    # axs[1].plot(resfit_dict['freq_data']/1e6, S21_data_mag_dB, '.C0', markersize=3, label='data add IQ Mixer calib')
    # axs[1].plot(resfit_dict['freq_data']/1e6, S21_data_norm_mag_dB, 'k', linewidth=2, label='fit add IQ Mixer calib')
    # axs[1].plot(fres/1e6, S21_data_norm_mag_dB[i_fres], '*C1', markersize=10, label='fres add IQ Mixer calib')
    
    # baseline_fit, S21_norm_fit = calc_res_fit(resfit_dict['freq_data_raw'], resfit_dict['resObj_raw'], resfit_dict['fit_params_raw'])
    # baseline_data, S21_norm_data = calc_res_fit(resfit_dict['freq_data_raw'], resfit_dict['resObj_raw'], resfit_dict['fit_params_raw'])
    baseline_data, S21_norm_fit = calc_res_fit(resfit_dict['freq_data_raw'], resfit_dict['resObj_raw'], resfit_dict['fit_params_raw'])
    fres = resfit_dict['fres_raw']
    i_fres = np.argmin(np.abs(fres - resfit_dict['freq_data_raw']))
    S21_data_norm = (resfit_dict['I_data_raw'] + 1j*resfit_dict['Q_data_raw']) / baseline_data
    S21_data_norm_mag_dB = 20 * np.log10(np.abs(S21_data_norm))
    S21_fit_norm_mag_dB = 20 * np.log10(np.abs(S21_norm_fit))
    axs[0].plot(np.real(S21_data_norm), np.imag(S21_data_norm), '.g', markersize=3, label='data_raw')
    axs[0].plot(np.real(S21_norm_fit), np.imag(S21_norm_fit), 'r', linewidth=2, label='fit_raw')
    axs[0].plot(np.real(S21_norm_fit[i_fres]), np.imag(S21_norm_fit[i_fres]), '*b', markersize=10, label='fres_raw')
    axs[1].plot(resfit_dict['freq_data_raw']/1e6, S21_data_norm_mag_dB, '.g', markersize=3, label='data_raw')
    axs[1].plot(resfit_dict['freq_data_raw']/1e6, S21_fit_norm_mag_dB, 'r', linewidth=2, label='fit_raw')
    axs[1].plot(fres/1e6, S21_fit_norm_mag_dB[i_fres], '*b', markersize=10, label='fres_raw')
    
    # S21_data_mag_dB = 20 * np.log10(np.sqrt(np.square(data_dict['I_data'])+np.square(data_dict['Q_data'])))
    # axs[0].plot(data_dict['I_data'], data_dict['Q_data'], '.g', markersize=3, label='data raw')
    # axs[0].plot(np.real(S21_data_norm), np.imag(S21_data_norm), 'r', linewidth=2, label='fit raw')
    # axs[0].plot(np.real(S21_data_norm[i_fres]), np.imag(S21_data_norm[i_fres]), '*b', markersize=10, label='fres raw')
    # axs[1].plot(resfit_dict['freq_data_raw']/1e6, S21_data_mag_dB, '.g', markersize=3, label='data raw')
    # axs[1].plot(resfit_dict['freq_data_raw']/1e6, S21_data_norm_mag_dB, 'r', linewidth=2, label='fit raw')
    # axs[1].plot(fres/1e6, S21_data_norm_mag_dB[i_fres], '*b', markersize=10, label='fres raw')
    
    axs[0].legend(fontsize=14)
    axs[1].legend(fontsize=14)
    str_params1 = '(After add calib)fres = %.1f MHz:  Qi=%.2e,  Qc=%.2e,  Qr=%.2e,  '%(resfit_dict['fres']/1e6, resfit_dict['Qi'], resfit_dict['Qc'], resfit_dict['Qr'])
    str_params1 += ' (without calib)fres = %.1f MHz:  Qi=%.2e,  Qc=%.2e,  Qr=%.2e,  '%(resfit_dict['fres_raw']/1e6, resfit_dict['Qi_raw'], resfit_dict['Qc_raw'], resfit_dict['Qr_raw'])
    plt.text(0.5, 0.95, str_params1, fontsize=12, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
    figtitle = '%s:  %dmK  %ddBm'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), int(meas_dict['pwr@device']))
    fig.text(0.5, 1, figtitle, fontsize=12, horizontalalignment='center', verticalalignment='top')
    
    save_plots(main_dict, meas_dict, 'fit_res')
    plt.close('all')
    
    return 


def take_calib(main_dict, meas_dict, resfit_dict, i_res):

    fres = resfit_dict['fres']
    Qi = resfit_dict['Qi']
    Qr = resfit_dict['Qr']
    
    meas_dict['freq_center'] = fres
    meas_dict['span'] = fres / Qr / 10
    meas_dict['f_cutoff'] = fres / 2 / Qr
    meas_dict['freq_start'] = main_dict['freq_center'][i_res] - meas_dict['span'] / 2
    meas_dict['freq_stop'] = main_dict['freq_center'][i_res] + meas_dict['span'] / 2
    
    meas_dict = check_FM_params(meas_dict)
    # main_dict = init_measurement(main_dict)
    # meas_dict = set_atten_and_pwr(main_dict, meas_dict)
    SR560_dict = set_SR560(meas_dict['mode'])
    time.sleep(5)
    SG382 = init_synthesizer_SG382(main_dict, meas_dict)
    SG382.write('ENBR 1')
    task, meas_dict = init_daq(meas_dict)
    time.sleep(30)
    print('Calib measurement (square FM): ')
    print('Center freq: %.3f MHz, FM deviation = %.2f kHz'%(meas_dict['freq_center']/1e6, meas_dict['span']/1e3))
    
    daq_data = daq_get_data(main_dict, meas_dict, task)
    # daq_data[1] = daq_data[1]*-1 # we take the inverse of Q data, because the resonance fit fails if we don't
    # IQ mixer calibration here

    data_dict = {'I_data':daq_data[0], 'Q_data':daq_data[1], 'Mod_data':daq_data[2]}
    task.close()
    SG382.write('ENBR 0')
    return main_dict, meas_dict, SR560_dict, data_dict


def calc_calib(data_dict, meas_dict):
    
    I_data = data_dict['I_data']
    Q_data = data_dict['Q_data']
    Mod_data = data_dict['Mod_data']
    calib_dict={}
    def func_sin(i, a, b):
        return np.sin((i-a)*b)
    
    a_ini, b_ini = 0, 2*np.pi/meas_dict['Npts_period']
    popt, pcov = curve_fit(func_sin, np.arange(len(Mod_data)), Mod_data, p0=[a_ini, b_ini])
    mod_sin = func_sin(np.arange(len(Mod_data)), popt[0], popt[1])
    
    # plt.figure()
    # plt.plot(daq_data[2], '.')
    # plt.plot(func_sin(np.arange(len(daq_data[2])), popt[0], popt[1]))
    
    Mod_low_i = np.where(mod_sin < -0.8)[0]
    Mod_high_i = np.where(mod_sin > 0.8)[0]
    Mod_i = np.concatenate([Mod_low_i, Mod_high_i])

    freq_data = np.ones(len(Mod_low_i)+len(Mod_high_i))
    freq_data[:len(Mod_low_i)] = meas_dict['freq_start']*np.ones(len(Mod_low_i))
    freq_data[len(Mod_low_i):] = meas_dict['freq_stop']*np.ones(len(Mod_high_i))
    
    # calib start
    IQ_low_mean = np.mean(I_data[Mod_low_i]) + 1j * np.mean(Q_data[Mod_low_i])
    IQ_high_mean = np.mean(I_data[Mod_high_i]) + 1j * np.mean(Q_data[Mod_high_i])
    angle = np.angle(IQ_high_mean - IQ_low_mean) - np.pi/2
    IQ_low_mean_new = IQ_low_mean * np.exp(-1j*angle)
    IQ_high_mean_new = IQ_high_mean * np.exp(-1j*angle)
    print(angle*180/np.pi)
    dIQ_mag = np.sqrt((IQ_high_mean.real - IQ_low_mean.real)**2 + (IQ_high_mean.imag - IQ_low_mean.imag)**2)
    dIQ_vs_df = dIQ_mag / meas_dict['span']
    calib_dict['I_data_raw']=I_data[Mod_i]
    calib_dict['Q_data_raw']=Q_data[Mod_i]
    calib_dict['Mod_data_raw']=Mod_data[Mod_i]
    calib_dict['freq_data_raw']=freq_data
    calib_dict['IQ_mean_raw']=[IQ_low_mean, IQ_high_mean]
    calib_dict['angle_raw']=angle
    calib_dict['dIQ_vs_df_raw']=dIQ_vs_df
    # calib end
    
    I_data_calib, Q_data_calib = main_calib_IQ_ACdata_d(I_data[Mod_i], Q_data[Mod_i], freq_data, IQ_calibfile_path=IQ_calibfileAC_path)
    IQ_low_mean = np.mean(I_data_calib[:len(Mod_low_i)]) + 1j * np.mean(Q_data_calib[:len(Mod_low_i)])
    IQ_high_mean = np.mean(I_data_calib[len(Mod_low_i):]) + 1j * np.mean(Q_data_calib[len(Mod_low_i):])

    angle = np.angle(IQ_high_mean - IQ_low_mean) - np.pi/2
    IQ_low_mean_new = IQ_low_mean * np.exp(-1j*angle)
    IQ_high_mean_new = IQ_high_mean * np.exp(-1j*angle)
    print(angle*180/np.pi)
    
    dIQ_mag = np.sqrt((IQ_high_mean.real - IQ_low_mean.real)**2 + (IQ_high_mean.imag - IQ_low_mean.imag)**2)
    dIQ_vs_df = dIQ_mag / meas_dict['span']
    
    data_dict['IQ_mean'] = [IQ_low_mean, IQ_high_mean]
    data_dict['angle'] = angle
    data_dict['dIQ_vs_df'] = dIQ_vs_df
    
    calib_dict['I_data']=I_data_calib
    calib_dict['Q_data']=Q_data_calib
    calib_dict['Mod_data']=Mod_data[Mod_i]
    calib_dict['freq_data']=freq_data
    calib_dict['IQ_mean']=[IQ_low_mean, IQ_high_mean]
    calib_dict['angle']=angle
    calib_dict['dIQ_vs_df']=dIQ_vs_df
    
    return calib_dict



def calc_calib_calib(data_dict, meas_dict):
    
    I_data = data_dict['I_data']
    Q_data = data_dict['Q_data']
    Mod_data = data_dict['Mod_data']
    
    IQ_low_mean = np.mean(I_data[:data_dict['len_Mod_low']]) + 1j * np.mean(Q_data[:data_dict['len_Mod_low']])
    IQ_high_mean = np.mean(I_data[data_dict['len_Mod_low']:]) + 1j * np.mean(Q_data[data_dict['len_Mod_low']:])

    angle = np.angle(IQ_high_mean - IQ_low_mean) - np.pi/2
    IQ_low_mean_new = IQ_low_mean * np.exp(-1j*angle)
    IQ_high_mean_new = IQ_high_mean * np.exp(-1j*angle)
    print(angle*180/np.pi)
    
    dIQ_mag = np.sqrt((IQ_high_mean.real - IQ_low_mean.real)**2 + (IQ_high_mean.imag - IQ_low_mean.imag)**2)
    dIQ_vs_df = dIQ_mag / meas_dict['span']
    
    data_dict['IQ_mean'] = [IQ_low_mean, IQ_high_mean]
    data_dict['angle'] = angle
    data_dict['dIQ_vs_df'] = dIQ_vs_df

    return data_dict


def init_noise_measurement(main_dict, meas_dict):
    SG382 = init_synthesizer_SG382(main_dict, meas_dict)
    SG382.write('ENBR 1')
    task, meas_dict = init_daq(meas_dict)
    time.sleep(5)
    return SG382, task
    


def noise_main(main_dict, meas_dict, SG382, task):

    time.sleep(5)
    
    # if init == True:
    #     SG382 = init_synthesizer_SG382(main_dict, meas_dict)
    #     SG382.write('ENBR 1')
    #     task, meas_dict = init_daq(meas_dict)
    #     time.sleep(5)
    
    print('onres noise measurement: ')
    print('Freq: %.3f MHz'%(meas_dict['freq_center']/1e6))
    SG382.write('FREQ %d'%(meas_dict['freq_center']))
    print('frequency set to %.6f MHz'%(float(SG382.query('FREQ?'))*1e-6))

    daq_onres = daq_get_data(main_dict, meas_dict, task)
    # daq_onres[1] = daq_onres[1]*-1 # we take the inverse of Q data, because the resonance fit fails if we don't
    I_ondata_calib, Q_ondata_calib = main_calib_IQ_ACdata_d(daq_onres[0], daq_onres[1], meas_dict['freq_center']*np.ones(len(daq_onres[0])), IQ_calibfile_path=IQ_calibfileAC_path)
    
    print('offres noise measurement: ')
    offres_freq = meas_dict['freq_center'] + main_dict['df_onoff_res']
    print('Freq: %.3f MHz'%(offres_freq/1e6))
    SG382.write('FREQ %d'%(offres_freq))
    print('frequency set to %.6f MHz'%(float(SG382.query('FREQ?'))*1e-6))
      
    daq_offres = daq_get_data(main_dict, meas_dict, task)
    # daq_offres[1] = daq_offres[1]*-1 # we take the inverse of Q data, because the resonance fit fails if we don't
    I_offdata_calib, Q_offdata_calib = main_calib_IQ_ACdata_d(daq_offres[0], daq_offres[1], offres_freq*np.ones(len(daq_offres[0])), IQ_calibfile_path=IQ_calibfileAC_path)
    
    # noise_dict = {'I_onres':I_ondata_calib, 'Q_onres':Q_ondata_calib, 'I_offres':I_offdata_calib, 'Q_offres':Q_offdata_calib}
    
    noise_dict = {'I_onres_raw':daq_onres[0], 'Q_onres_raw':daq_onres[1], 'I_offres_raw':daq_offres[0], 'Q_offres_raw':daq_offres[1],
                  'I_onres':I_ondata_calib, 'Q_onres':Q_ondata_calib, 'I_offres':I_offdata_calib, 'Q_offres':Q_offdata_calib}

    return noise_dict

def rotate_noise(noise_dict, calib_dict):  
    '''
    To identify the dissipation and frequency directions of the noise, 
    the noise blob needs to be rotated so the dissipation direction 
    corresponds to the imaginary part and the frequency direction is 
    the real part. To do that we use the 2 calibration points.
    The line joining the 2 pts is parallel to the tangent of the IQ circle: 
    It is the frequency direction. So we rotate the noise blob in order to have this 
    line parallel to the real axis.
    Before rotating we center the noise blob on 0 so the rotation does not shift it.
    '''
    noise_onres = noise_dict['I_onres_raw'] + 1j * noise_dict['Q_onres_raw']
    noise_offres = noise_dict['I_offres_raw'] + 1j * noise_dict['Q_offres_raw']
    noise_onres_centered = noise_onres - np.mean(noise_onres)
    noise_onres_centered_rot = noise_onres_centered * np.exp(-1j*calib_dict['angle_raw'])
    noise_offres_centered = noise_offres - np.mean(noise_offres)
    noise_offres_centered_rot = noise_offres_centered * np.exp(-1j*calib_dict['angle_raw'])
    calib_mean_centered = calib_dict['IQ_mean_raw'] - np.mean(noise_onres)
    calib_mean_centered_rot = calib_mean_centered * np.exp(-1j*calib_dict['angle_raw'])

    noise_dict['noise_onres_centered_raw'] = noise_onres_centered
    noise_dict['noise_onres_centered_rot_raw'] = noise_onres_centered_rot
    noise_dict['noise_offres_centered_raw'] = noise_offres_centered
    noise_dict['noise_offres_centered_rot_raw'] = noise_offres_centered_rot
    noise_dict['noise_onres_mean_centered_raw'] = 0 # The noise blob is now centered on 0
    noise_dict['noise_onres_mean_centered_rot_raw'] = 0
    noise_dict['calib_mean_centered_raw'] = calib_mean_centered
    noise_dict['calib_mean_centered_rot_raw'] = calib_mean_centered_rot
    
    
    noise_onres = noise_dict['I_onres'] + 1j * noise_dict['Q_onres']
    noise_offres = noise_dict['I_offres'] + 1j * noise_dict['Q_offres']
    noise_onres_centered = noise_onres - np.mean(noise_onres)
    noise_onres_centered_rot = noise_onres_centered * np.exp(-1j*calib_dict['angle'])
    noise_offres_centered = noise_offres - np.mean(noise_offres)
    noise_offres_centered_rot = noise_offres_centered * np.exp(-1j*calib_dict['angle'])
    calib_mean_centered = calib_dict['IQ_mean'] - np.mean(noise_onres)
    calib_mean_centered_rot = calib_mean_centered * np.exp(-1j*calib_dict['angle'])

    noise_dict['noise_onres_centered'] = noise_onres_centered
    noise_dict['noise_onres_centered_rot'] = noise_onres_centered_rot
    noise_dict['noise_offres_centered'] = noise_offres_centered
    noise_dict['noise_offres_centered_rot'] = noise_offres_centered_rot
    noise_dict['noise_onres_mean_centered'] = 0 # The noise blob is now centered on 0
    noise_dict['noise_onres_mean_centered_rot'] = 0
    noise_dict['calib_mean_centered'] = calib_mean_centered
    noise_dict['calib_mean_centered_rot'] = calib_mean_centered_rot
    return noise_dict



def plot_calib_rotation(calib_dict, main_dict, meas_dict):
    IQ_mean_new_calib = np.array(calib_dict['IQ_mean']) * np.exp(-1j*calib_dict['angle'])
    plt.figure(figsize=(17, 10))
    
    plt.plot(calib_dict['I_data'], calib_dict['Q_data'],  '.', color='y', label='calib data')
    plt.plot(calib_dict['IQ_mean'][0].real, calib_dict['IQ_mean'][0].imag, '*r', markersize=10)
    plt.plot(calib_dict['IQ_mean'][1].real, calib_dict['IQ_mean'][1].imag, '*c', markersize=10)
    plt.plot([calib_dict['IQ_mean'][0].real, calib_dict['IQ_mean'][1].real], [calib_dict['IQ_mean'][0].imag, calib_dict['IQ_mean'][1].imag], 'g', linewidth=2)
    plt.plot(IQ_mean_new_calib[0].real, IQ_mean_new_calib[0].imag, '*r', markersize=10)
    plt.plot(IQ_mean_new_calib[1].real, IQ_mean_new_calib[1].imag, '*c', markersize=10)
    plt.plot([IQ_mean_new_calib[0].real, IQ_mean_new_calib[1].real], [IQ_mean_new_calib[0].imag, IQ_mean_new_calib[1].imag], 'g', linewidth=2)
    figtitle = '%s:  %dmK  %ddBm  angle: %.2frad'%(main_dict['dev_name'], int(meas_dict['FPtemp_start']*1000), int(meas_dict['pwr@device']), calib_dict['angle'])
    plt.title(figtitle, fontsize=12)
    
    plt.xticks(color='k', size=20)
    plt.yticks(color='k', size=20)
    plt.xlabel("I  [V]", fontsize = 22)
    plt.ylabel('Q  [V]', fontsize = 22)
    plt.grid(visible=True, which='both', color='0.65', linestyle='-')
    plt.gca().set_aspect('equal', 'box')
    plt.show() 
    
    save_plots(main_dict, meas_dict, 'calib_rotation')
    time.sleep(5)
    plt.close('all')
      

def calc_dff(noise_dict, calib_dict, main_dict, meas_dict):
    # noise_dict['data_noise_onres_df_f'] = np.zeros([noise_dict['N_pwr'], noise_dict['N_onres'], noise_dict['N_meas'], noise_dict['Npts_noise']], dtype=complex)        
    # df_noise = noise_dict['freq_calib'][:,1] - noise_dict['freq_calib'][:,0]
    # dS21_cal = noise_dict['calib_mean'][:,1] - noise_dict['calib_mean'][:,0]
    # dS21_cal_abs = np.abs(dS21_cal)
    noise_dict['data_noise_onres_df_f_raw'] = noise_dict['noise_onres_centered_rot_raw'] / calib_dict['dIQ_vs_df_raw'] / meas_dict['freq_center']
    noise_dict['data_noise_offres_df_f_raw'] = noise_dict['noise_offres_centered_rot_raw'] / calib_dict['dIQ_vs_df_raw'] / meas_dict['freq_center']
    
    noise_dict['data_noise_onres_df_f'] = noise_dict['noise_onres_centered_rot'] / calib_dict['dIQ_vs_df'] / meas_dict['freq_center']
    # noise_dict['data_noise_offres_df_f'] = noise_dict['noise_offres_centered_rot'] / calib_dict['dIQ_vs_df'] / meas_dict['freq_center']
    noise_dict['data_noise_offres_df_f'] = noise_dict['noise_offres_centered_rot'] / calib_dict['dIQ_vs_df'] / (meas_dict['freq_center']+main_dict['df_onoff_res'])
    return noise_dict 


def calc_Sdff(meas_dict, noise_dict, Sdff_dict, i):
    '''
    Calculates the PSD of the noise in df/fres units.
    If usewelch we use the builtin welch function which averages all the noise PSD chunks.
    If not usewelch, we use a loop to compute each noise PSD and discard the chunks which are too far from the noise PSD median.
    Pxx_f_min is the minimum audio frequency requested. It is similar to N_avg but more intuitive.
    '''
    fs = meas_dict['DAQ_freq']
    nperseg = int(meas_dict['DAQ_Npts']/50) # 2024/04/02 FD changed from 10 to 50 to get less noise
    print('averaging factor = 50')
    f, PSD_onres_diss = sig.welch(np.real(noise_dict['data_noise_onres_df_f_raw']), fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
    _, PSD_onres_freq = sig.welch(np.imag(noise_dict['data_noise_onres_df_f_raw']), fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
    _, PSD_offres_freq = sig.welch(np.imag(noise_dict['data_noise_offres_df_f_raw']), fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
    _, PSD_offres_diss = sig.welch(np.real(noise_dict['data_noise_offres_df_f_raw']), fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
    Sdff_dict['f'] = f
    Sdff_dict['PSD_onres_diss_raw'+str(i)] = PSD_onres_diss
    Sdff_dict['PSD_onres_freq_raw'+str(i)] = PSD_onres_freq
    Sdff_dict['PSD_offres_diss_raw'+str(i)] = PSD_offres_diss
    Sdff_dict['PSD_offres_freq_raw'+str(i)] = PSD_offres_freq
    
    f, PSD_onres_diss = sig.welch(np.real(noise_dict['data_noise_onres_df_f']), fs=fs, window='hann', nperseg=nperseg, return_onesided = True)   
    _, PSD_onres_freq = sig.welch(np.imag(noise_dict['data_noise_onres_df_f']), fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
    _, PSD_offres_freq = sig.welch(np.imag(noise_dict['data_noise_offres_df_f']), fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
    _, PSD_offres_diss = sig.welch(np.real(noise_dict['data_noise_offres_df_f']), fs=fs, window='hann', nperseg=nperseg, return_onesided = True) 
    Sdff_dict['f'] = f
    Sdff_dict['PSD_onres_diss'+str(i)] = PSD_onres_diss
    Sdff_dict['PSD_onres_freq'+str(i)] = PSD_onres_freq
    Sdff_dict['PSD_offres_diss'+str(i)] = PSD_offres_diss
    Sdff_dict['PSD_offres_freq'+str(i)] = PSD_offres_freq
    return Sdff_dict

    
def average_Sdff(Sdff_dict, N_noise):
    PSD_onres_diss_avg=0; PSD_onres_freq_avg=0; PSD_offres_diss_avg=0;PSD_offres_freq_avg=0
    for i in range(N_noise):
        PSD_onres_diss_avg += Sdff_dict['PSD_onres_diss_raw'+str(i)]
        PSD_onres_freq_avg += Sdff_dict['PSD_onres_freq_raw'+str(i)]
        PSD_offres_diss_avg += Sdff_dict['PSD_offres_diss_raw'+str(i)]
        PSD_offres_freq_avg += Sdff_dict['PSD_offres_freq_raw'+str(i)]
        
    Sdff_dict['PSD_onres_diss_avg_raw'] = PSD_onres_diss_avg / N_noise
    Sdff_dict['PSD_onres_freq_avg_raw'] = PSD_onres_freq_avg / N_noise
    Sdff_dict['PSD_offres_diss_avg_raw'] = PSD_offres_diss_avg / N_noise
    Sdff_dict['PSD_offres_freq_avg_raw'] = PSD_offres_freq_avg / N_noise
    
    PSD_onres_diss_avg=0; PSD_onres_freq_avg=0; PSD_offres_diss_avg=0;PSD_offres_freq_avg=0
    for i in range(N_noise):
        PSD_onres_diss_avg += Sdff_dict['PSD_onres_diss'+str(i)]
        PSD_onres_freq_avg += Sdff_dict['PSD_onres_freq'+str(i)]
        PSD_offres_diss_avg += Sdff_dict['PSD_offres_diss'+str(i)]
        PSD_offres_freq_avg += Sdff_dict['PSD_offres_freq'+str(i)]
        
    Sdff_dict['PSD_onres_diss_avg'] = PSD_onres_diss_avg / N_noise
    Sdff_dict['PSD_onres_freq_avg'] = PSD_onres_freq_avg / N_noise
    Sdff_dict['PSD_offres_diss_avg'] = PSD_offres_diss_avg / N_noise
    Sdff_dict['PSD_offres_freq_avg'] = PSD_offres_freq_avg / N_noise
    
    return Sdff_dict
    
def plot_Sdff_ini(meas_dict):
    lines = []
    fig1, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
    fig1.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
    lines.append(axs[0].loglog([], [], linewidth=2, color='C0', linestyle='-', label = 'onres: freq'))
    lines.append(axs[0].loglog([], [], linewidth=2, color='C1', linestyle='-', label = 'onres: diss'))
    lines.append(axs[0].loglog([], [], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff'))
    lines.append(axs[1].loglog([], [], linewidth=2, color='C0', linestyle='-', label = 'offres: freq'))
    lines.append(axs[1].loglog([], [], linewidth=2, color='C1', linestyle='-', label = 'offres: diss'))
    lines.append(axs[1].loglog([], [], linewidth=2, color='k', linestyle=':', label = 'resonance cutoff'))

    # ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 10)
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
    plt.pause(0.05)
    plt.show(block=False)
    return fig1, axs, lines
    

def plot_Sdff_update(meas_dict, Sdff_dict, fig1, axs, lines, ylims=[1e-20, 1e-14], saveplots=False):
    lines[0][0].set_data(Sdff_dict['f'], np.abs(Sdff_dict['PSD_onres_freq_avg']))
    lines[1][0].set_data(Sdff_dict['f'], np.abs(Sdff_dict['PSD_onres_diss_avg']))
    lines[2][0].set_data([meas_dict['f_cutoff'], meas_dict['f_cutoff']], ylims)
    lines[3][0].set_data(Sdff_dict['f'], np.abs(Sdff_dict['PSD_offres_freq_avg']))
    lines[4][0].set_data(Sdff_dict['f'], np.abs(Sdff_dict['PSD_offres_diss_avg']))
    lines[5][0].set_data([meas_dict['f_cutoff'], meas_dict['f_cutoff']], ylims)
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize = 14)
    axs[0].relim()
    axs[0].set_ylim(ylims)
    axs[0].autoscale_view()  
    axs[1].relim()
    axs[1].set_ylim(ylims)
    axs[1].autoscale_view()  
    plt.pause(0.05)
    plt.draw()
    plt.show(block=False) 
    return 


def plot_Sdff(Sdff_dict, main_dict, meas_dict, ylims=[1e-20, 1e-14]):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
    fig.subplots_adjust(bottom=0.09, top=0.92, right=0.75, left=0.09)
    # axs[0].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_onres_freq_avg']), linewidth=2, color='C0', linestyle=':', label = 'onres: freq')
    # axs[0].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_onres_diss_avg']), linewidth=2, color='C1', linestyle=':', label = 'onres: diss')
    # axs[0].loglog([meas_dict['f_cutoff'], meas_dict['f_cutoff']], ylims, linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
    # axs[1].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_offres_freq_avg']), linewidth=2, color='C0', linestyle=':', label = 'offres: freq')
    # axs[1].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_offres_diss_avg']), linewidth=2, color='C1', linestyle=':', label = 'offres: diss')
    # axs[1].loglog([meas_dict['f_cutoff'], meas_dict['f_cutoff']], ylims, linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
    
    axs[0].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_onres_freq_avg_raw']), linewidth=2, color='r', linestyle='-', label = 'onres: freq')
    axs[0].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_onres_diss_avg_raw']), linewidth=2, color='b', linestyle='-', label = 'onres: diss')
    axs[0].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_onres_freq_avg']), linewidth=2, color='r', linestyle=':', label = 'calib onres: freq')
    axs[0].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_onres_diss_avg']), linewidth=2, color='b', linestyle=':', label = 'calib onres: diss')
    axs[0].loglog([meas_dict['f_cutoff'], meas_dict['f_cutoff']], ylims, linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
    axs[1].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_offres_freq_avg_raw']), linewidth=2, color='r', linestyle='-', label = 'offres: freq')
    axs[1].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_offres_diss_avg_raw']), linewidth=2, color='b', linestyle='-', label = 'offres: diss')
    axs[1].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_offres_freq_avg']), linewidth=2, color='r', linestyle=':', label = 'calib offres: freq')
    axs[1].loglog(Sdff_dict['f'], np.abs(Sdff_dict['PSD_offres_diss_avg']), linewidth=2, color='b', linestyle=':', label = 'calib offres: diss')
    axs[1].loglog([meas_dict['f_cutoff'], meas_dict['f_cutoff']], ylims, linewidth=2, color='k', linestyle=':', label = 'resonance cutoff')
    
    
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
    # axs[0].set_ylim(ylims)
    axs[0].autoscale_view()  
    axs[1].relim()
    # axs[1].set_ylim(ylims)
    axs[1].autoscale_view()  
    
    save_plots(main_dict, meas_dict, 'Sdff')
    time.sleep(5)
    plt.close('all')
    return 

def fit_tau_Nqp(f, SNqp, f_range=(1e1, 1e5)):
    def minimize_func(params, f, SNqp):
        omega = 2*np.pi*f
        return (4 * params['tau'] * params['Nqp']) / (1 + (params['tau'] * omega)**2) + params['Cte'] - SNqp
    i_min = np.argmin(np.abs(f_range[0] - f))
    i_max = np.argmin(np.abs(f_range[1] - f))
    f = f[i_min:i_max]
    fit_dict = []
    SNqp_trimmed = SNqp[i_min:i_max]
    # SNqp_trimmed = remove_spikes(SNqp_trimmed)
    fit_params = lf.Parameters()
    fit_params.add('Cte', value=1e-3, min=0, max=500)
    fit_params.add('tau', value=1e-5, min=1e-7, max=1e-3)
    fit_params.add('Nqp', value=1e4, min=1e2, max=1e8)
    mini = lf.Minimizer(minimize_func, fit_params, nan_policy='omit', fcn_args=(f, SNqp_trimmed))
    out1 = mini.minimize(method='basinhopping')
    # out1 = mini.minimize(method='Nelder')
    out2 = mini.minimize(method='Nelder', params=out1.params)
    fit_dict.append(out2.params)
    return fit_dict
    

xi = lambda omega, T: (hbar_eV  * omega)/(2* kb_eV * T)


# k2 calculation
def get_k2(omega, T, delta_0, N0):
    # delta = get_delta(delta_0, T)
    k2 = 1/(2*N0*delta_0) * (1 + np.sqrt(2*delta_0/(np.pi*kb_eV*T)) * np.exp(-xi(omega, T)) * sp.iv(0, xi(omega, T)))
    return k2


def calc_SNqp(fres, T, Sdff_dict, delta_0, V):
    # k2 = get_k2(Sdff_dict['PSD_onres_freq']*2*np.pi, T*1e-3, delta_0, N0)
    # SNqp = 4 * V**2 * Sdff_dict['PSD_onres_freq'] / k2**2 
    k2 = get_k2(fres*2*np.pi, T*1e-3, delta_0, N0)
    SNqp = 4 * V**2 * Sdff_dict['PSD_onres_freq_avg'] / k2**2 
    return SNqp
    

def SNqp_fit_func(f, tau_qp, Nqp, Cte):
    omega = 2*np.pi*f
    return (4 * tau_qp * Nqp) / (1 + (tau_qp * omega)**2) + Cte


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

        
def save_plots(main_dict, meas_dict, filedir_name):
    fitres_dir = main_dict['plot_dir'] + filedir_name + '/'
    os.makedirs(fitres_dir, exist_ok=True)
    filename = fitres_dir + 'fitres_%sMHz'%meas_dict['freq_center_str'] + '_pwr_%sdBm'%meas_dict['pwr@device_str'] + '_T_%smK'%(int(meas_dict['FPtemp_start']*1000))
    # print('Saving %s at %s MHz with %s dBm pwr'%(filedir_name, meas_dict['freq_center'], meas_dict['pwr@device']))
    plt.savefig(filename + '.svg', dpi=mydpi, bbox_inches = 'tight')
    plt.savefig(filename + '.png', dpi=mydpi, bbox_inches = 'tight')
    plt.savefig(filename + '_tiny.png', dpi=int(mydpi/4.5), bbox_inches = 'tight')
    plt.savefig(filename + '.pdf', dpi=mydpi, bbox_inches = 'tight')
            


# def load_MB_file():

#     SNqp = calc_SNqp(FP_temp[i], Sdff_dict)
#     SNqp_fit_params = fit_tau_Nqp(Sdff_dict['f'], SNqp)
    
    
#     # filename_h5 = '233049_IQmixer_res_287p79MHz_275mK.h5'
#     # filename_h5 = '233049_IQmixer_res_287p80MHz_300mK.h5'
#     # filename_h5 = '233049_IQmixer_res_287p81MHz_320mK.h5'
#     filename_h5 = '233049_IQmixer_res_287p77MHz_350mK.h5'
    
#     with h5py.File(main_dict['h5_dir'] + filename_h5, 'r') as hf:  
#         # hf[gr_source]
#         gr_source = 'source_%s'%(main_dict['source'])
#         gr_FPT = list(hf[gr_source].keys())[0]
#         gr_pwr = list(hf[gr_source][gr_FPT].keys())[0]
#         gr_res = 'res_scan'
#         gr_Sdff = 'Sdff'
#         hf[gr_source][gr_FPT][gr_pwr][gr_res].keys()
#         Sdff_dataset = hf[gr_source][gr_FPT][gr_pwr][gr_res][gr_Sdff]['PSD_onres_freq'][()]
#         f_dataset = hf[gr_source][gr_FPT][gr_pwr][gr_res][gr_Sdff]['f'][()]
    
#     Sdff_dict = {'PSD_onres_freq':Sdff_dataset, 'f':f_dataset}

    
    
# def plot_SNqp(meas_dict, noise_dict, vna_dict, SNqp_fit=None, saveplots=False, ylims=None):
#     '''
#     Plot SNqp. If the fit of Sdff (SNqp_fit) is given, it is plotted, if not, only raw SNqp is plotted.
#     The first 2 values of SNqp are skipped, the first is always f=0Hz and not relevant (its SNqp corresponding value is the same as SNqp[1])
#     the second (index = 1) seems to be biased and lower than what it should be. I think Scipy has options to determine how it should be treated, 
#     but it is easier to just discard it. Taking a longer set of data is not so hard.
#     '''
    
#     alpha = 1
#     V=3224
#     # f0 = np.array(vna_dict['fres']) + np.array(vna_dict['df'])
#     fig = plt.figure(figsize=(16, 9))
#     plt.subplots_adjust(bottom=0.09, top=0.94, right=0.66, left=0.08)
#     plt.xticks(color='k', size=20)
#     plt.yticks(color='k', size=20)
#     plt.xlabel("Frequency [Hz]", fontsize = 22)
#     plt.ylabel(r'$S_{Nqp}$ [$Hz^{-1}$]', fontsize = 22)
    
#     # currdens, p_int, n_int, field_int = calc_phys_params(0, level, meas_dict['device_pwr'], vna_dict['qr'][i], vna_dict['qc'][i], f0[i])
#     # str_params1 = 'fres=%.2f MHz, Qi=%.2e, Qc=%.2e'%(f0[i]/1e6, vna_dict['qi'][i], vna_dict['qc'][i])
#     # str_params1 += '\n'
#     # str_params1 += '$P_{internal}$=%.2e W (%d dBm), $n_{photons}$=%.2e' %(p_int, 10*np.log10(p_int*1e3), n_int)
#     # str_params1 += '\n'
#     # str_params1 += 'E=%.2e V/m, J=%.2e A/m$^2$'%(field_int, currdens)
    
#     # Sdff_diss = noise_dict['Pxx_f'][2:]**SNqp_fit['a0'].value * Sdff_fit['b0'].val + Sdff_fit['c0'].val
#     SNqp_freq = SNqp_fit_func(Sdff_dict['f'][2:], SNqp_fit_params[0]['tau'].value, SNqp_fit_params[0]['Nqp'].value, SNqp_fit_params[0]['Cte'].value)
#     str_params2 = r'GR Noise fit: $\tau_{qp}=%.1f \mu s$, $n_{qp}=%.2e \mu m^{-3}$'%((SNqp_fit_params[0]['tau'].value)*1e6, (SNqp_fit_params[0]['Nqp'].value)/V)
#     # Sdff_TLS = noise_dict['Pxx_f'][2:]**Sdff_fit['a1'].val * Sdff_fit['b1'].val
#     # ymin = np.min([Sdff_diss, Sdff_freq, Sdff_TLS])*0.5
#     # ymax = np.max([Sdff_diss, Sdff_freq, Sdff_TLS])*2
#     # alpha = 0.5

#     plt.loglog(Sdff_dict['f'][2:], SNqp[2:], '-C0', linewidth=2, alpha=alpha)#, label = str_params1)
#     # plt.loglog(noise_dict['Pxx_f'][2:], noise_dict['Pxx_diss'][2:], '-C1', linewidth=2, alpha=alpha, label = 'diss')   
#     # plt.loglog(noise_dict['Pxx_f'][2:], Sdff_TLS, '--k', linewidth=2, label='Noise fit (TLS): %.2e $f^{%.2f}$'%(Sdff_fit['b1'].val, Sdff_fit['a1'].val))
#     # plt.loglog(noise_dict['Pxx_f'][2:], SNqp_freq, '--k', linewidth=2, label=r'GR Noise fit: $\tau_{qp}=%.1f \mu s$, $n_{qp}=%.2e \mu m^{-3}$'%((SNqp_fit[i]['tau'].value)*1e6, (SNqp_fit[i]['Nqp'].value)/V))
#     plt.loglog(Sdff_dict['f'][2:], SNqp_freq, '--ko', linewidth=2,  markerfacecolor='C0', markevery=0.1, label=str_params2)

#         # plt.loglog(noise_dict['Pxx_f'][2:], Sdff_freq, 'C0', linewidth=2, label=r'Noise fit (Freq): Noise fit (TLS) + %.2f $\times$ Noise fit (diss)'%(Sdff_fit['c1'].val))
#     # else:
#     #     ax = plt.gca()
#     #     [ymin, ymax] = ax.get_ylim()
#     # if ylims:
#     #     [ymin, ymax] = [ylims[0], ylims[1]]
#     # plt.loglog([vna_dict['f_cutoff'][i], vna_dict['f_cutoff'][i]], [ymin, ymax], 'k', label='resonator cutoff')

#     plt.xticks(color='k', size=18)
#     plt.yticks(color='k', size=18)
#     plt.xlabel("Frequency [Hz]", fontsize = 20)
#     plt.ylabel(r'$S_{Nqp}$ [$Hz^{-1}$]', fontsize = 20)
#     # plt.legend(fontsize = 14)
#     plt.legend(loc="upper right", bbox_to_anchor=[0.99, 0.94], bbox_transform = plt.gcf().transFigure, ncol=1, shadow=True, fancybox=True, fontsize=12)
#     plt.grid(visible=True, which='both', color='0.65', linestyle='-')
#     # plt.text(0.5, 0.97, meas_dict['str_title_res'], fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
#     # plt.text(0.5, 0.92, str_params1, fontsize=14, horizontalalignment='center', verticalalignment='center', transform = plt.gcf().transFigure)
#     plt.xlim([Sdff_dict['f'][2], Sdff_dict['f'][-1]])
#     # plt.ylim([ymin, ymax])
#     # plt.show()  
#     # if saveplots:
#     #     plotdir = meas_dir + 'Plots/'
#     #     filenames = [meas_dict['dev_name'] + '_PSD_SNqp']
#     #     save_plots(plotdir, filenames, nums)
#     #     return filenames
    
    
    
    
    



        



