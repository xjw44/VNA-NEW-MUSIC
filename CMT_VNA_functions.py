# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:23:50 2019

@author: F Defrance
"""

import win32com.client # Allows communication via COM interface
import numpy as np
import matplotlib.pyplot as plt
from Hardware_Control import SIM921_RT as GRT
from Hardware_Control import read_lakeshore_sensors as LS
from Hardware_Control import SIM960_PID
import tkinter as tk
from tkinter import filedialog
import h5py, os, time, queue
from Hardware_Control import setSMASwitch
from Hardware_Control import PS2520G_pwr_supply as PS


instrument = 'S2VNA'
#Instantiate COM client
try:
	app = win32com.client.Dispatch(instrument + ".application")
except:
	print("Error establishing COM server connection to " + instrument + ".")
	print("Check that the VNA application COM server was registered")
	print("at the time of software installation.")
	print("This is described in the VNA programming manual.")
	input("\nPress Enter to Exit Program\n")
	exit()

#Wait up to 20 seconds for instrument to be ready
if app.Ready == 0:
    print("Instrument not ready! Waiting...")
    for k in range (1, 21):
        time.sleep(1)
        if app.Ready != 0:
            break
        print("%d" % k)

# If the software is still not ready, cancel the program
if app.Ready == 0:
	print("Error, timeout waiting for instrument to be ready.")
	print("Check that VNA is powered on and connected to PC.")
	input("\nPress Enter to Exit Program\n")
	exit()
else:
    print("Instrument ready! Continuing...")


def restart_app():
    instrument = 'S2VNA'
    #Instantiate COM client
    try:
        app = win32com.client.Dispatch(instrument + ".application")
        return app
    except:
        print("Error establishing COM server connection to " + instrument + ".")
        print("Check that the VNA application COM server was registered")
        print("at the time of software installation.")
        print("This is described in the VNA programming manual.")
        input("\nPress Enter to Exit Program\n")
        exit()

    #Wait up to 20 seconds for instrument to be ready
    if app.Ready == 0:
        print("Instrument not ready! Waiting...")
        for k in range (1, 21):
            time.sleep(1)
            if app.Ready != 0:
                break
            print("%d" % k)

    # If the software is still not ready, cancel the program
    if app.Ready == 0:
        print("Error, timeout waiting for instrument to be ready.")
        print("Check that VNA is powered on and connected to PC.")
        input("\nPress Enter to Exit Program\n")
        exit()
    else:
        print("Instrument ready! Continuing...")
    

def setup_VNA(Ch, Tr):
    app.scpi.system.preset()
    time.sleep(0.2)
    app.scpi.GetCALCulate(Ch).GetPARameter(Tr).define = 'S21'
    app.scpi.GetCALCulate(Ch).GetPARameter(Tr).select()
    # Set the computer signal as trigger source 
    # (required to have the VNA wait until the end of the measurement before accepting other commands)
    app.scpi.TRIGger.SEQuence.Source = "BUS" 
    # disable display
    app.scpi.display.enable = False
    
def close_VNA():
    app.scpi.TRIGger.SEQuence.Source = "Internal" 
    app.scpi.display.enable = True



def Measure_S21(f_center, f_span, dev_name, cryo_chan, f_step=100, power=-55, Navg=1, BW=1e3, init=False):
    """
    CopperMountain TR1300/1 S21 measurement
    f_start, f_stop, f_step and BW in Hz
    Power in dBm
    dev_name the name of the device under test
    cryo_chan the number of the cryostat channel used (1-4)
    Navg the VNA averaging factor
    BB_temp the temperature of the Black body in K
    """
    Ch = 1 # VNA channel
    Tr = 1 # Channel trace
    f_start = np.round(f_center - f_span/2)    
    f_stop = np.round(f_center + f_span/2)

    if init: setup_VNA(Ch, Tr)
    app.scpi.GetSOURce(Ch).power.level.immediate.amplitude = power
    app.scpi.GetSENSe(Ch).bandwidth.resolution = BW
    app.scpi.GetSENSe(Ch).AVERage.COUNt = Navg
    if Navg > 1:
        app.scpi.GetSENSe(Ch).AVERage.STATe = True
    else:
        app.scpi.GetSENSe(Ch).AVERage.STATe = False
   
    Npts_max = 16001 # max num of pts the VNA can scan
    Npts = int(f_span/f_step)
    Nscans = int(np.ceil(Npts/Npts_max)) # Num of needed iterations
    Nfc = len(str(f_center))
    str1 = str(np.round(f_center*10**(5-Nfc)))
    f_center_MHz_str = str1[:Nfc-6] + 'p' + str1[Nfc-6:]
    scan  = {}
    scan['Date'] = time.strftime('%Y%m%d')
    for i in ('dev_name', 'cryo_chan', 'f_start', 'f_stop', 'f_step', 'f_span', 'f_center', 
              'f_center_MHz_str', 'Npts', 'power', 'BW', 'Navg'):
        scan[i] = locals()[i]
    
    #scan_str, scan = get_scan_params(scan, status = 'start')
    #print(scan_str)
    print('\nMeasurement starting...\n')
    
    # Perform S21 measurement 
    # Do multiple iterations and copy S21 and freq data in python variables
    freq, S21_re, S21_im = [], [], []
    pt_start = 0
    for i in range(Nscans):
        pt_end = min(pt_start + Npts_max-1, Npts)
        Npts_tmp = pt_end - pt_start + 1
        f_span_tmp = (Npts_tmp-1) * f_step
        f_start_tmp = f_start + pt_start * f_step
        f_stop_tmp = f_start_tmp + f_span_tmp
        print('Scanning %.3f MHz to %.3f MHz'%(f_start_tmp/1e6, f_stop_tmp/1e6))
        pt_start = pt_end + 1
        app.scpi.GetSENSe(Ch).frequency.start = f_start_tmp
        app.scpi.GetSENSe(Ch).frequency.stop = f_stop_tmp
        app.scpi.GetSENSe(Ch).sweep.points = Npts_tmp
        for avg_i in range(Navg):
            app.scpi.trigger.sequence.single()
        
        app.scpi.GetCALCulate(Ch).selected.format = 'SCOM' #'MLIN' #'MLOG' 
        S21 = app.SCPI.GetCALCulate(Ch).SELected.DATA.FDATa
        freq_tmp = app.SCPI.GetSENSe(Ch).FREQuency.DATA
        freq.extend([round(a) for a in freq_tmp])
        S21_re.extend(S21[0::2])
        S21_im.extend(S21[1::2])
    
    S21data = {'freq':freq, 'S21_re':S21_re, 'S21_im':S21_im}
    return S21data
    
        
def plot_data_file(file_name = None):
    """
    Plot the data in the filename 
    Use the filename specified in args or open a dialog window to let the user choose
    """
    if not file_name:
        root = tk.Tk()
        root.withdraw()
        file_name = filedialog.askopenfilename (title = "Select file to open") # Open a dialog window to select the file to plot
    data = np.loadtxt(file_name)
    freq = np.array([a[0] for a in data])
    S21_re = np.array([a[1] for a in data])
    S21_im = np.array([a[2] for a in data])
    plt.figure()
    plt.plot(freq/1e6, 10*np.log10(S21_re**2 + S21_im**2))
    plt.xlabel('Frequency [MHz]')
    plt.ylabel(r'$|S21|^2$ [dBm]')
    plt.grid()
    
    
def get_scan_params(scan, status='start'):
    """
    Get all the useful parameters and save them in params dictionary
    if status = start get most of the params including timestamp at the start
    if status = stop get the measurement time and temperature at the end of the scan
    """
    if status == 'start':
        time_yyyymmdd = time.strftime('%Y%m%d')
        time_hhmm = time.strftime('%Hh%M')
        timestamp_start = int(time.time())
        FPtemp_start = float(GRT.GetTemp())*1000
        BBtemp_start = LS.get_diode_temp(5, verb=0)
        
        args = (time_yyyymmdd + ' ' + time_hhmm, scan['dev_name'], scan['cryo_chan'], 
        scan['power'], scan['BW'], scan['Navg'], scan['Npts'], scan['f_start']/1e6, 
        scan['f_stop']/1e6, scan['f_step'], FPtemp_start, BBtemp_start)
        print(scan)
        scan_str = '''CopperMountain TR1300/1 VNA: S21 measurement
        Scan parameters:
        Date: {:s}
        Device name: {:s}
        Cryostat channel: {:d} (channel used for the device: [1 - 4])
        Start frequency: {:.3f} MHz
        Stop frequency: {:.3f} MHz
        Frequency step: {:.0f} Hz
        Focal plane temperature @ start: {:.2f} mK
        Black body temperature @ start: {:.3f} K
        Num of frequency points: {:.0f}
        Output power: {:.0f} dBm
        IF bandwidth: {:.0f} Hz
        Averaging factor: {:.0f}
        '''.format(*args)
        
        for i in ('time_yyyymmdd', 'time_hhmm', 'timestamp_start', 'FPtemp_start', 'BBtemp_start', 'scan_str'):
            scan[i] = locals()[i]
        
    if status == 'stop':
        scan_time = int(time.time()) - scan['timestamp_start']
        FPtemp_stop = float(GRT.GetTemp())*1000
        BBtemp_stop = LS.get_diode_temp(5, diode_type = 'DT670', verb=0)
        for i in ('scan_time', 'FPtemp_stop', 'BBtemp_stop'):
            scan[i] = locals()[i]
        args = (BBtemp_stop, FPtemp_stop, scan_time)
        scan_str = '''
    Black body temperature @ end: {:.3f} K
    Focal plane temperature @ end: {:.2f} mK
    Scan time: {:d} seconds'''.format(*args)
    
    return scan_str, scan




def check_args(meas_args, power_sweep=False):
    """
    Function to format, check, and initialize the measurement parameters
    in meas_args dictionary, the required parameters are f_start, f_stop, dev_name, and cryo_chan,
    corresponding respectively to the VNA sweep start and stop frequencies, the name of the devices
    for each channel, and the numbers of the cryostat channels used (1, 2, 3, and/or 4). 
    It is possible to scan up to 4 devices (channels) in a row. 
    Additional parameters such as f_step, BW, power, Navg, corresponding respectively to the frequency step, 
    the IF bandwidth, the power and the number of averaged measurements for the VNA are optionnal but can also be
    set in meas_args dictionary. If not defined, the function will set them to the default values:
    f_step = 100 Hz, BW = 1000 Hz, power = -55 dBm, Navg = 1. 
    3 sweeps are possible: power sweep, focal plane temperature sweep, and black body temperature sweep.
    FP temp and BB temp sweeps are automatically detected when an array is assigned to BB_temp or FP_temp
    in meas_args. For power sweep, it is necessary to assign an array to power in meas_args and set the
    variable power_sweep to True. When only assigning an array of power values to power without setting power_sweep=True
    means that the different power values correspond to the measurement power of each of the scan/devices.
    When setting only a single value to BB_temp and/or FP_temp means that we want a single scan at this/these
    temperature(s).
    Examples of meas_args values:
        meas_args = {'f_start':[[50e6, 60e6, 1e9], [50e6, 1e9]], 'f_stop':[[51e6, 100e6, 1.1e9], [100e6, 1.1e9]], 'FP_temp':None, 'power':[-10, -30],
                 'dev_name':['Be123456', 'Be765432'], 'cryo_chan':[1, 4], 'f_step':[[200, 200, 1000], [200, 1000]], 'BW':1000, "BB_temp":None}
        meas_args = {'f_start':[[50e6, 60e6, 1e9], [50e6, 1e9]], 'f_stop':[[51e6, 100e6, 1.1e9], [100e6, 1.1e9]], 'FP_temp':270, 'power':[-10, -30],
                 'dev_name':['Be123456', 'Be765432'], 'cryo_chan':[1, 4], 'f_step':[[200, 200, 1000], [200, 1000]], 'BW':1000, "BB_temp":[3.5, 4, 5, 6, 7]}    
        
    """
    Nchan = 0
    Nres = []
    Npower = 0
    mode_BBtemp = "passive" # choice between passive, single, and sweep
    mode_FPtemp = "passive" # choice between passive, single, and sweep
    mode_power = "single"
    mode = ""

    required_keys = ['f_start', 'f_stop', 'dev_name', 'cryo_chan']
    if not all([a in meas_args for a in required_keys]):
        print("Error! Missing required item in meas_args...")
        return       
        
    default_vals = {'f_step':100, 'power':-55, 'Navg':1, 'BW':1e3, 'BB_temp':None, 'FP_temp':None}
    for key in default_vals:
        if not key in meas_args:
            meas_args.update([(key, default_vals[key])])
    
    for arg in meas_args:
        if np.isscalar(meas_args[arg]): 
            meas_args[arg] = [meas_args[arg]]
        if hasattr(meas_args[arg], "__len__"):
            meas_args[arg] = list(meas_args[arg])

    if len(meas_args['f_start']) == len(meas_args['f_stop']):
        Nchan = len(meas_args['f_start'])
        for i in range(Nchan):
            if not hasattr(meas_args['f_start'][i], "__len__"):
                meas_args['f_start'][i] = [meas_args['f_start'][i]]
            if not hasattr(meas_args['f_stop'][i], "__len__"):
                meas_args['f_stop'][i] = [meas_args['f_stop'][i]]
            if len(meas_args['f_start'][i]) == len(meas_args['f_stop'][i]):
                Nres.append(len(meas_args['f_start'][i]))
            else: 
                print('Error! f_start and f_stop do not have the same number of values and/or dimensions')
                return
            for k in range(Nres[-1]):
                if hasattr(meas_args['f_start'][i][k], "__len__") or hasattr(meas_args['f_stop'][i][k], "__len__"):
                    print('Error! Too many dimensions in f_start and/or f_stop')
                    return
    
    if (hasattr(meas_args['cryo_chan'], "__len__") and hasattr(meas_args['dev_name'], "__len__")):
        if len(meas_args['cryo_chan']) == len(meas_args['dev_name']):
            if not Nchan == len(meas_args['cryo_chan']): 
                print("Error! f_start, f_stop, cryo_chan, dev_name do not all have the same length")
                return
    else: 
        print("Error! cryo_chan and dev_name do not have the same length")
        return
    
    list_keys = ['f_step', 'power', 'Navg', 'BW']
    if power_sweep: list_keys = ['f_step', 'Navg', 'BW']
    
    for key in list_keys:
        if len(meas_args[key]) != Nchan:
            meas_args[key] = meas_args[key] * Nchan
        for i in range(Nchan):
            if not hasattr(meas_args[key][i], "__len__"):
                meas_args[key][i] = [meas_args[key][i]] * Nres[i]
            if not len(meas_args[key][i]) == Nres[i]:
                print('Error! f_step, power, Navg, or BW does not have the correct shape!')
                return
        if not len(meas_args[key]) == Nchan:
            print('Error! f_step, power, Navg, or BW does not have the correct shape!')
            return
            
    BW_ok = np.array([1, 3, 5, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4])
    power_ok = np.array([-55, 0])
    cryo_chan_ok = np.array([1, 2, 3, 4])

    for a in range(Nchan):
        for b in range(Nres[a]):
            if meas_args['f_start'][a][b] >= meas_args['f_stop'][a][b]:
                temp = meas_args['f_start'][a][b]
                meas_args['f_start'][a][b] = meas_args['f_stop'][a][b]
                meas_args['f_stop'][a][b] = temp
                print("Error! A value of f_start is greater than the corresponding f_stop value, the 2 values have been inverted")
            if meas_args['BW'][a][b] not in BW_ok:
                BW_i = np.argmin(abs(meas_args['BW'][a][b] - BW_ok))
                meas_args['BW'][a][b] = BW_ok[BW_i]
                print("Error! The VNA bandwidth was set to an incorrect value. It has been corrected and set to the closest valid value: %d Hz"%BW_ok[BW_i])
            if not power_sweep:
                if meas_args['power'][a][b] < power_ok[0]:
                    meas_args['power'][a][b] = power_ok[0]
                    print("Error! The required VNA power is too low. It was set to the lowest available value: %d dBm"%power_ok[0])
                elif meas_args['power'][a][b] > power_ok[1]:
                    meas_args['power'][a][b] = power_ok[1]
                    print("Error! The required VNA power is too high. It was set to the highest recommanded value: %d dBm"%power_ok[1])
                
        if (meas_args['cryo_chan'][a] not in cryo_chan_ok):
            print("Error! cryo_chan not set to an avalaible value: 1, 2, 3, or 4")
            return
    
    if (meas_args['BB_temp'] != None):
        if len(meas_args['BB_temp']) == 1:
            mode_BBtemp = "single"
        elif len(meas_args['BB_temp']) > 1:
            mode_BBtemp = "sweep"
        else:
            print("Error! Incorrect dimension for BB_temp")
            return
        for BB in meas_args['BB_temp']:
            if hasattr(BB, "__len__"):
                print("Error! BB_temp array has more than 1 dimension")
                return
        
        meas_args['BB_temp'] = np.sort(meas_args['BB_temp'])
        for temp in meas_args['BB_temp']:
            if temp < 3.4:
                print("Error! BB temp requested too low (%.2f K). Lowest reachable temperature is 3.4 K"%temp)
                return
            elif temp > 15:
                print("Error! BB temp requested probably too high (%.2f K). Highest recommanded temperature is 12 K"%temp)
                return
                
    if meas_args['FP_temp'] != None:
        if len(meas_args['FP_temp']) == 1:
            mode_FPtemp = "single"
        elif len(meas_args['FP_temp']) > 1:
            mode_FPtemp = "sweep"
        for FP in meas_args['FP_temp']:
            if hasattr(FP, "__len__"):
                print("Error! FP_temp array has more than 1 dimension")
                return
        
        meas_args['FP_temp'] = np.sort(meas_args['FP_temp'])
        if meas_args['FP_temp'][0] > 100: meas_args['FP_temp'] = meas_args['FP_temp'] / 1e3
        for temp in meas_args['FP_temp']:
            if temp < 0.240:
                print("Error! FP temp requested too low (%.2f K). Lowest reachable temperature is 0.240 K"%temp)
                return
            if temp > 0.600:
                print("Error! FP temp requested probably too high (%.2f K). Highest recommanded temperature is 0.5 K"%temp)
                return
    
    if power_sweep: 
        mode_power = "sweep" # choice between single and sweep
        if len(meas_args['power']) > 1:
            Npower = len(meas_args['power'])
            meas_args['power'] = np.sort(meas_args['power'])
            for pwr in meas_args['power']:
                if pwr < power_ok[0] or pwr > power_ok[1]:
                    print("Error! Some of the required VNA power values are either too high or too low. Please set the power values within [-55, 0] dBm.")
                    return
                if hasattr(pwr, "__len__"):
                    print("Error! power array has more than 1 dimension")
                    return
        else: 
            print("Error! mode_power set to sweep but length of power array <= 1")
            return

    else: 
        mode_power = "single"
        Npower = 1
        
    if mode_BBtemp == 'sweep' and mode_FPtemp == 'sweep':
        print("Error! impossible to have BBtemp sweep and FPtemp sweep at the same time")
        return 

    meas_args['mode_BBtemp'] = mode_BBtemp
    meas_args['mode_FPtemp'] = mode_FPtemp
    meas_args['mode_power'] = mode_power
    meas_args['Npower'] = Npower
    meas_args['Nchan'] = Nchan
    meas_args['Nres'] = Nres
    
    # for arg in meas_args.values():
    #     print(arg)
    
    dev_str = ""
    for i in range(Nchan):
        dev_str += "      channel %d  -->  "%meas_args['cryo_chan'][i] + meas_args['dev_name'][i] + "\n" 
    
    mode_str = "\n    * Measurement mode: \n"
    if meas_args['mode_BBtemp'] == 'sweep': mode_str += "      Black Body temperature sweep:\n      %s K\n"%str(list(meas_args['BB_temp']))
    if meas_args['mode_FPtemp'] == 'sweep': mode_str += "      Focal Plane temperature sweep:\n      %s mK\n"%str(list(1e3*meas_args['FP_temp']))
    if meas_args['mode_power'] == 'sweep': mode_str += "      VNA power sweep:\n      %s dBm\n"%str(list(meas_args['power']))
    if meas_args['mode_BBtemp'] == 'single': mode_str += "      Black Body set to single temperature: %.3f K \n"%meas_args['BB_temp']
    if meas_args['mode_FPtemp'] == 'single': mode_str += "      Focal Plane set to single temperature: %d mK \n"%(meas_args['FP_temp']*1e3)
    if meas_args['mode_BBtemp'] == 'passive': mode_str += "      Black Body not heated \n"
    if meas_args['mode_FPtemp'] == 'passive': mode_str += "      Focal Plane not heated \n"

    VNA_str = "\n    * VNA Parameters: \n"    
    f_start_MHz = [[ 1e-6 * i for i in k ] for k in meas_args['f_start']]
    f_stop_MHz = [[ 1e-6 * i for i in k ] for k in meas_args['f_stop']]
    
    for i in range(Nchan):
        VNA_str += '      -- Channel %d --    \n'%meas_args['cryo_chan'][i]
        VNA_str += '      Start frequencies: {} MHz \n'.format(str(f_start_MHz[i]))
        VNA_str += '      Stop frequencies: {} MHz \n'.format(str(f_stop_MHz[i]))
        VNA_str += '      Frequency step: {} Hz \n'.format(str(meas_args['f_step'][i]))
        VNA_str += '      IF bandwidth: {} Hz \n'.format(str(meas_args['BW'][i]))
        VNA_str += '      Averaging factor: {} \n'.format(str(meas_args['Navg'][i]))
        if not power_sweep: 
            VNA_str += '      Power: {} dBm \n'.format(str(meas_args['power'][i]))
                
    meas_str = "\n    ---   Summary of the measurfement parameters   --- \n\n"
    meas_str += dev_str + mode_str + VNA_str
    
    if meas_args['mode_BBtemp'] == 'sweep':
        mode = 'BBtemp_sweep'
    elif meas_args['mode_FPtemp'] == 'sweep':
        mode = 'FPtemp_sweep'
    elif meas_args['mode_power'] == 'sweep':
        mode = 'power_sweep'
    else: 
        mode = 'single_scan'
    meas_args['mode'] = mode    
    
    print(meas_str)
    str_in = input("\nEnter y to continue or any key to abort: \n")
    if str_in == "Y" or str_in == "y":
        time_yyyymmdd = time.strftime('%Y%m%d')
        meas_args['time_yyyymmdd'] = time_yyyymmdd
        return meas_args
    else: return        
        
    
    
    
    
def take_data(meas_args):
    init = True
    for i_chan in range(meas_args['Nchan']):
        newh5 = True
        filename = None
        dev_name = meas_args['dev_name'][i_chan]
        cryo_chan = meas_args['cryo_chan'][i_chan]
        SetSMASwitch(meas_args['cryo_chan'][i_chan])
        for i_res in range(meas_args['Nres'][i_chan]):
            f_start = meas_args['f_start'][i_chan][i_res]
            f_stop = meas_args['f_stop'][i_chan][i_res]
            f_step = meas_args['f_step'][i_chan][i_res]
            BW = meas_args['BW'][i_chan][i_res]
            Navg = meas_args['Navg'][i_chan][i_res]
            if meas_args['mode_power'] == 'single':
                power = meas_args['power'][i_chan][i_res]
                scan, S21data = Measure_S21(f_start, f_stop, dev_name, cryo_chan, f_step=f_step, 
                                         power=power, Navg=Navg, BW=BW, init=init)
                init = False
                path_h5 = meas_args['data_dir'][i_chan]['data_h5']
                mode = meas_args['mode']
                filename = save_S21_h5(scan, S21data, path_h5, mode, filename=filename, newh5=newh5)
                newh5 = False
            elif meas_args['mode_power'] == 'sweep':
                for i_pwr in range(meas_args['Npower']):
                    scan, S21data = Measure_S21(f_start, f_stop, dev_name, cryo_chan, f_step=f_step, 
                                             power=power, Navg=Navg, BW=BW, init=init)
                    init = False
                    path_h5 = meas_args['data_dir'][i_chan]['data_h5']
                    mode = meas_args['mode']
                    filename = save_S21_h5(scan, S21data, path_h5, mode, filename=filename, newh5=newh5)
                    newh5 = False



def save_S21_h5(scan, S21data, path_h5, mode, filename=None, newh5=True):
    
    def open_h5(path_h5, filename):
        fullname = path_h5 + filename
        if os.path.isfile(fullname) and newh5:
            i = 1
            while os.path.isfile(fullname):
                filename = filename[:-6] + '_%.2d.h5'%i
                fullname = path_h5 + filename
                i += 1
        return filename
    
    if path_h5[-1] != '/': path_h5 += '/'    
    
    if mode == "single_scan":
        if not filename:
            filename = scan['dev_name'] + '_' + str(round(scan['power'])) + 'dBm_' + str(round(scan['FPtemp_start']*1e3)) +'mK_00.h5'
            filename = open_h5(path_h5, filename)
        with h5py.File(path_h5 + filename, 'a') as hf:
            gr_res = scan['f_center_MHz_str'] + 'MHz'
            gr_data = 'data'
            g1 = hf.create_group(gr_res)
            g2 = g1.create_group(gr_data)
            g2.create_dataset('freq',data=S21data['freq'])
            g2.create_dataset('S21_re',data=S21data['S21_re'])
            g2.create_dataset('S21_im',data=S21data['S21_im'])
            scan_ID = (a for a in scan.keys())
            for attr in scan_ID:
        #        print(attr, params[attr])
                g2.attrs[attr] = scan[attr]

    elif mode == "BBtemp_sweep":
        if not filename:
            filename = scan['dev_name'] + '_' + str(round(scan['FPtemp_start']*1e3)) +'mK_00.h5'
            filename = open_h5(path_h5, filename)
        with h5py.File(path_h5 + filename, 'a') as hf:        
            gr_BBT = 'BBtemp_%.3fK'%scan['BBtemp_start']
            gr_res = scan['f_center_MHz_str'] + 'MHz'
            gr_power = str(scan['power']) + 'dBm'
            gr_data = 'data'
            g1 = hf.create_group(gr_BBT)
            g2 = g1.create_group(gr_power)
            g3 = g2.create_group(gr_res)
            g4 = g3.create_group(gr_data)
            g4.create_dataset('freq',data=S21data['freq'])
            g4.create_dataset('S21_re',data=S21data['S21_re'])
            g4.create_dataset('S21_im',data=S21data['S21_im'])
            scan_ID = (a for a in scan.keys())
            for attr in scan_ID:
        #        print(attr, params[attr])
                g4.attrs[attr] = scan[attr]

    elif mode == "FPtemp_sweep":
        if not filename:
            filename = scan['dev_name'] + '_00.h5'
            filename = open_h5(path_h5, filename)
        with h5py.File(path_h5 + filename, 'a') as hf:
            gr_FPT = 'FPtemp_%dmK'%(scan['FPtemp_start']*1e3)
            gr_res = scan['f_center_MHz_str'] + 'MHz'
            gr_power = str(scan['power']) + 'dBm'
            gr_data = 'data'
            g1 = hf.create_group(gr_FPT)
            g2 = g1.create_group(gr_power)
            g3 = g2.create_group(gr_res)
            g4 = g3.create_group(gr_data)
            g4.create_dataset('freq',data=S21data['freq'])
            g4.create_dataset('S21_re',data=S21data['S21_re'])
            g4.create_dataset('S21_im',data=S21data['S21_im'])
            scan_ID = (a for a in scan.keys())
            for attr in scan_ID:
        #        print(attr, params[attr])
                g4.attrs[attr] = scan[attr]

    elif mode == "power_sweep":
        if not filename:
            filename = scan['dev_name'] + '_' + str(round(scan['FPtemp_start']*1e3)) +'mK_00.h5'
            filename = open_h5(path_h5, filename)
        with h5py.File(path_h5 + filename, 'a') as hf:
            gr_res = scan['f_center_MHz_str'] + 'MHz'
            gr_power = str(scan['power']) + 'dBm'
            gr_data = 'data'
            g1 = hf.create_group(gr_power)
            g2 = g1.create_group(gr_res)
            g3 = g2.create_group(gr_data)
            g3.create_dataset('freq',data=S21data['freq'])
            g3.create_dataset('S21_re',data=S21data['S21_re'])
            g3.create_dataset('S21_im',data=S21data['S21_im'])
            scan_ID = (a for a in scan.keys())
            for attr in scan_ID:
        #        print(attr, params[attr])
                g3.attrs[attr] = scan[attr]
    else: 
        print('Error! Wrong mode name')
        return
    
    return filename
        
    
 
def create_folder(meas_args, root_dir):
    """
    For different kinds of measurements: Black body temperature sweep, 
    power sweep, temperature sweep, or single measurement
    create the appropriate folder where to save the data and log files
    """
    # root_dir = 'C:/Users/kids/Documents/YellowCryostatSweepData/'
    # root_dir = 'C:/Users/DELL E7270/Desktop/'
    if root_dir[-1] != '/': root_dir = root_dir + '/'
    if not os.path.exists(root_dir):
        print('Error! root_dir does not exist!')
        return
        
    if meas_args['mode_BBtemp'] == 'sweep':
        dev_dir = 'BBtemp_sweep'
    elif meas_args['mode_FPtemp'] == 'sweep':
        dev_dir = 'FPtemp_sweep'
    elif meas_args['mode_power'] == 'sweep':
        dev_dir = 'Power_sweep'
    else: 
        dev_dir = 'Single_scans'
    
    dir_names = ['data_txt', 'log_txt', 'data_h5', 'log_h5']
    dir_list = []
    for i in range(meas_args['Nchan']):
        new_dir = root_dir + meas_args['dev_name'][i] + '/' + meas_args['time_yyyymmdd'] + '/' + dev_dir
        if dev_dir != 'Single_scans':
            dir_i = 0           
            while os.path.exists(new_dir + '_%.2d'%dir_i):
                print(new_dir + '_%.2d'%dir_i + ' already exists') 
                dir_i += 1 
            new_dir += '_%.2d'%dir_i
            
        dir_dict = {}
        for k in range(len(dir_names)):
            str_dir = new_dir + '/Raw_data/' + dir_names[k] + '/'
            dir_dict[dir_names[k]] = str_dir
            if not os.path.exists(str_dir):
                print(str_dir + ' has been created') 
                os.makedirs(str_dir)
        dir_list.append(dir_dict)
            
    meas_args['data_dir'] = dir_list
    return meas_args
        
        

# def save_txt(data, params, save_dirs, BBT_PID_data = None):
#     # Save the data and log files 

#     if not BBT_PID_data:
#         filename = (params['time_yyyymmdd'] + '_' + params['dev_name'] + '_' + str(round(params['f_center']/1e3)) 
#         + 'kHz_' + str(params['power']) + 'dBm_' + str(round(params['FPtemp_start'])) + 'mK_' + 'BBT_%.3fK'%params['BBtemp_start'])
#         np.savetxt(save_dirs['data_dir'] + filename + '.txt', np.c_[data['freq'], data['S21_re'], data['S21_im']], 
#         delimiter='\t', fmt=('%d', '%.8f', '%.8f'))
#         with open(save_dirs['logs_dir'] + filename + '_log.txt', "w") as logfile:
#             logfile.write(params['params_str'])
            
#     if BBT_PID_data:
#         filename = (params['time_yyyymmdd'] + '_' + params['dev_name'])       
#         np.savetxt(save_dirs['logs_dir'] + filename + '_BBT_PID_data.dat', np.c_[BBT_PID_data['x_time'], BBT_PID_data['y_volt'], BBT_PID_data['y_BBT']], 
#                    fmt=['%d', '%.3f', '%.3f'], delimiter='\t', header='Time [s]    BB temp [K]   1KOhms Heater voltage [V]')




    
#hf = h5py.File(filename+'.h5', 'r')
#
#hf.keys()


#temp = np.arange(250, 490, 20)
# f_center, f_span, dev_name, cryo_chan, f_step, power, Navg, BW, BB_temp = 75e6, 2e6, 'Betest_code', 1, 100, -55, 1, 1000, 0
#f_center, f_span, dev_name, cryo_chan, f_step, power, Navg, BW, BB_temp = 75e6, 50e6, 'Be190807', 1, 200, -55, 1, 1000, 0

# save_dirs = create_folder(dev_name, mode = 'temp_sweep')
#params, data = Measure_S21(50e6, 55e6, 'NIST', 1, f_step=150)
# save_files(data, params, save_dirs)
# Measure_S21(f_center, f_span, dev_name, cryo_chan, f_step=f_step)
#getS21(50e6, 100e6, 'Be190807', 1, f_step=200, temp = temp, BW = 1000)

#f_start = [50e6, 100e6, 200e6, 500e6]
#f_stop = [100e6, 200e6, 500e6, 1.3e9]
##f_step = [200, 300, 400, 700]
#f_step = [500, 600, 700, 2000]
#
#getS21(f_start, f_stop, 'Be191113p1', 3, f_step=f_step, BW = 1000)
#
#
#temp = np.arange(250, 490, 25)
#temp = np.array([250, 270, 290, 315, 340, 370, 400, 430])
#f_start = np.array([46, 48, 50])*1e6
#f_stop = np.array([96, 100, 100])*1e6
#f_middle = (f_start+f_stop)/2
#f_step = np.round(f_middle/5e5)
#dev_name = ['Be190814p3bl','Be170227bl', 'Be190807tl']
#cryo_chan = [1, 2, 4]
#BB_temp = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
#FPtemp = 270
##SetSMASwitch(cryo_chan[i_chan])
##getS21_switch(f_start, f_stop, dev_name, cryo_chan, f_step=f_step, power=-55, Navg=1, BW=1e3, BB_temp=0, temp = temp, save_data=1, plot_data=0)
#getS21_switch(f_start, f_stop, dev_name, cryo_chan, f_step=f_step, power=-55, Navg=1, BW=1e3, BB_temp=BB_temp, temp = FPtemp, save_data=1, plot_data=0)


# f_start = np.array([61.5, 48, 50])*1e6
# f_stop = np.array([68, 100, 100])*1e6
# f_middle = (f_start+f_stop)/2
# f_step = np.round(f_middle/5e5)*2
# dev_name = ['Be190814p3bl','Be170227bl', 'Be190807tl']
# #dev_name = ['Be170227bl', 'Be190807tl']
# cryo_chan = [1, 2, 4]
# BB_temp = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
# FPtemp = 270
# #SetSMASwitch(cryo_chan[i_chan])
# #getS21_switch(f_start, f_stop, dev_name, cryo_chan, f_step=f_step, power=-55, Navg=1, BW=1e3, BB_temp=0, temp = temp, save_data=1, plot_data=0)
# getS21_switch(f_start, f_stop, dev_name, cryo_chan, f_step=f_step, power=-55, Navg=1, BW=1e3, BB_temp=BB_temp, temp = FPtemp, save_data=1, plot_data=0)





    

    
#meas_args = {'f_start':[[50e6, 60e6, 1e9], [50e6, 1e9]], 'f_stop':[[51e6, 61e6, 1.1e9], [52e6, 1.05e9]], 'FP_temp':None, 'power':[-50, -30],
#         'dev_name':['Be190814p3bl', 'Be190807tl'], 'cryo_chan':[1, 4], 'f_step':[[300, 300, 10000], [300, 10000]], 'BW':1000, "BB_temp":None}    
#root_dir = 'C:/Users/kids/Documents/YellowCryostatData/'
#meas_args = check_args(meas_args, power_sweep=False)    
#meas_args = create_folder(meas_args, root_dir)
#take_data(meas_args)
















