# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:56:12 2020

@author: Fabien Defrance

It is possible to read the temperature directly from Lakeshore instrument,
but the instrument just uses a look up table with linear interpolation between the points
and returns a temperature with format %.3f.
For the PID, it would be better to have a temperature with more digits (%.4f)
just to show variations smaller than 1 mK and adjust the PID in accordingly. 
Therefore, I created 2 lookup tables for the 2 diode models we have (DT470 and DT670).
For each temperature request, the function get_diode_temp reads the voltage of the diode,
then reads and interpolates the lookup table to obtain the temperature corresponding of the 
diode voltage. 
The execution of the function get_diode_temp only takes 9ms, and most of the time is taken
by the instrument to answer the request (so opening the lookup table file and interpolating it 
is super fast).
"""

import pyvisa
import numpy as np
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt


   
        
#def get_diode_temp(Ndiode, diode_type = 'DT670', verb=1):
#    """
#    Needs the diode channel number (Ndiode), and the diode type
#    (usually DT470 for our old diodes, and DT670 for the new ones)
#    """
#    if diode_type == 'DT670':
#        VT_data = np.loadtxt('DT-670_lookup.dat')
#    elif diode_type == 'DT470':
#        VT_data = np.loadtxt('DT-470_lookup.dat')
#    else:
#        print('Error! Unknown diode type... ')
#        return
#    
#    V = [a[0] for a in VT_data]
#    T = [a[1] for a in VT_data]
#    VTinterp = interp1d(V, T, kind='cubic')
#    #To prevent errors when the GPIB request collides with other requests from the LabView program
#    Vread = 0
#    while Vread < V[0] or Vread > V[-1]:
#        try:
#            if Vread != 0: 
#                print("Lakeshore diode %d readout error! Let's try again"%Ndiode)
#            LS218_1.write('SRDG? %d'%Ndiode)
#            LS218_1.read()
#            Vread = float(LS218_1.read())
#        except pyvisa.VisaIOError:
#            print("VisaIOError... Probably just a collision with LabView commands. Let's try again")
#        
#    BBT = VTinterp(Vread)
#    if verb: print("Sensor %d (%s) temperature: %.4f K"%(Ndiode, diode_type, BBT))
#    return float(BBT)


def get_diode_temp(Ndiode, verb=1):
    """Reads the temperature given by Lakeshore readout instrument"""
    rm = pyvisa.ResourceManager()    
    LS218_1 = rm.open_resource('GPIB0::1::INSTR')      
    flag_ok = False
    while not flag_ok:
#        while True:
        try:
            LS218_1.write('KRDG? %d'%Ndiode)
            BBT1 = float(LS218_1.read())
            LS218_1.write('KRDG? %d'%Ndiode)
            BBT2 = float(LS218_1.read())  
            if BBT1 == BBT2: flag_ok = True
            else: print("%.3f, %.3f"%(BBT1, BBT2) )
#            break
        except pyvisa.VisaIOError:
            print("VisaIOError... LS218. Let's try again!")
            time.sleep(1)
#
#        if BBT1 == BBT2: flag_ok = True
#        else: print("%.3f, %.3f"%(BBT1, BBT2) )
    if verb: print("Sensor %d temperature: %.4f K"%(Ndiode, BBT1))
#    print("try to close lakeshore diode")
    LS218_1.close()
    rm.close()
    return BBT1


def create_DT670_lookup():
    """ Function used to create DT-670 lookout file.
    It reads the lookout values already present inside Lakeshore readout instrument
    The way the instrument is set up right now, channel 4 is configured for DT-670"""
    V, T = [], []
    rm = pyvisa.ResourceManager()    
    LS218_1 = rm.open_resource('GPIB0::1::INSTR')  
    for i in np.arange(1,78):
        LS218_1.write('CRVPT? 4, %d'%i)
        str_dat = LS218_1.read().split(',')
        V.append(float(str_dat[0]))
        T.append(float(str_dat[1]))
        time.sleep(0.2)
    LS218_1.close()
    rm.close()
    np.savetxt('DT-670_lookup.dat', np.c_[V, T], header = 'Input [V], Temp [K]', fmt = ['%.5f', '%.2f'])
    
    
def create_DT470_lookup():
    """ Function used to create DT-470 lookout file.
    It reads the lookout values already present inside Lakeshore readout instrument
    The way the instrument is set up right now, channel 4 is configured for DT-470"""
    V, T = [], []
    rm = pyvisa.ResourceManager()    
    LS218_1 = rm.open_resource('GPIB0::1::INSTR')  
    for i in np.arange(1,89):
        LS218_1.write('CRVPT? 1, %d'%i)
        str_dat = LS218_1.read().split(',')
        V.append(float(str_dat[0]))
        T.append(float(str_dat[1]))
        time.sleep(0.2)
    LS218_1.close()
    rm.close()
    np.savetxt('DT-470_lookup.dat', np.c_[V, T], header = 'Input [V], Temp [K]', fmt = ['%.5f', '%.2f'])


def plot_lookup_files():
    """Plots the values in the lookup tables and the interpolated curves, for both sensors"""
    VT_470 = np.loadtxt('DT-470_lookup.dat')
    VT_670 = np.loadtxt('DT-670_lookup.dat')
    V470 = [a[0] for a in VT_470]
    T470 = [a[1] for a in VT_470]
    V670 = [a[0] for a in VT_670]
    T670 = [a[1] for a in VT_670]
    VT470interp = interp1d(V470, T470, kind='cubic')
    VT670interp = interp1d(V670, T670, kind='cubic')
    V470bis = np.linspace(V470[0], V470[-1], 1000)
    V670bis = np.linspace(V670[0], V670[-1], 1000)
    plt.figure()
    plt.plot(V470, T470, 'C0*', label = 'DT-470: Ref data')
    plt.plot(V470bis, VT470interp(V470bis), 'C0', label = 'DT-470: Interpolated')
    plt.plot(V670, T670, 'C1*', label = 'DT-670: Ref data')
    plt.plot(V670bis, VT670interp(V670bis), 'C1', label = 'DT-670: Interpolated')
    plt.xlabel('Diode Voltage [V]')
    plt.ylabel('Temperature [K]')
    plt.grid()
    plt.legend()




            
            
            
            