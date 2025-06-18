# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:23:50 2019

@author: F Defrance

Reads GRT values and converts the resistance in temperature.
Calibration files of the 4 GRTs are read and the resistance 
value is calculated by interpolating these data points.
The old version using Chebychev polynomial interpolation 
is still present, but commented at the end of the file. 
Interpolating the calibration data points is more accurate
which is why this solution is preferred.

UC stage (focal plane) -> GRT 29062
UC head -> GRT 28596
IC stage -> GRT 27523
IC head -> GRT 28618


"""

import numpy as np
# import math
import pyvisa
from scipy.interpolate import interp1d
import os.path
import time

root_dir = 'D:/Python_Scripts/temp_sensors_cal_files/'
num_gpib = 5


def R2T_interp(Rread, GRT_serial='29062', verb=0):
    """
    Calculates the temperature from resistance value
    Input: resistance [Ohm], GRT serial number and channel used on SIM921 instrument
    Output: temperature [K]    
    Reads the calibration file corresponding to the sensor and interpolates the data
    to calculate the temperature corresponding to the input resistance value.
    """
    filename = 'GRT_' + GRT_serial + '.dat'
    if os.path.isfile(root_dir + filename):
        GRT_data = np.loadtxt(root_dir + filename)
    else:
        print('Error! GRT Calibration file not found.')
        return
    
    T = [a[0] for a in GRT_data]
    R = [a[1] for a in GRT_data]
    if Rread < min(R) or Rread > max(R):
        print('Error! input resistance is outside of calibration range!')
        return
    GRTinterp = interp1d(np.log10(R), T, kind='cubic')
    Tread = GRTinterp(np.log10(Rread))
    if verb: print("GRT %s temperature: %.1f mK"%(GRT_serial, Tread*1e3))
    return float(Tread)  


def T2R_interp(Tread, GRT_serial='29062', verb=0):
    """
    Calculates the resistance from temperature value
    Input: temperature [K], GRT serial number and channel used on SIM921 instrument
    Output: resistance [Ohm]
    Reads the calibration file corresponding to the sensor and interpolates the data
    to calculate the resistance corresponding to the input temperature value.    
    """
    filename = 'GRT_' + GRT_serial + '.dat'
    if os.path.isfile(root_dir + filename):
        GRT_data = np.loadtxt(root_dir + filename)
    else:
        print('Error! GRT Calibration file not found.')
        return
    
    T = [a[0] for a in GRT_data]
    R = [a[1] for a in GRT_data]
    if Tread < min(T) or Tread > max(T):
        print('Error! input temperature is outside of calibration range!')
        return    
    GRTinterp = interp1d(T, np.log10(R), kind='cubic')
    Rread = 10**(GRTinterp(Tread))
    if verb: print("GRT %s resistance: %d Ohms"%(GRT_serial, round(Rread)))
    return float(Rread) 


def getRes(GRT_chan):
    """
    Reads the resistance of the temperature sensor
    Input: channel used on SIM921 instrument
    return: resistance [Ohm]
    Tests showed that about 1% of the time, a wrong value 
    '+8.157016E+02\r' is sent instead of the resistance value
    The quick and dirty solution is to remeasure the resistance
    """
    forbidden_resist = '+8.157016E+02\r'
    resist = forbidden_resist
    Ntry = 5
    i = 0
    while resist == forbidden_resist and i<Ntry:
        rm = pyvisa.ResourceManager()    
        inst = rm.open_resource('GPIB0::%d::INSTR'%num_gpib)
        inst.read_termination = '\n'
        inst.write_termination = '\n'
        inst.write('xyzzy')
        # time.sleep(0.01)
        inst.write('CONN %d,"xyzzy"'%GRT_chan)
        # time.sleep(0.01)
        inst.write("RVAL?")
        resist = inst.read()
        i += 1
    inst.close()
    rm.close()
    return float(resist)


def GetTemp(GRT_serial='29062', GRT_chan=1):
    """
    Reads GRT resistance from SIM921 and calculate temperature
    Input: GRT serial number and channel used on SIM921 instrument
    Output: temperature [K]
    The try/except structure prevents I/O collision in the instrument to stop the program
    All 4 GRTs can be read but since the one we are usually interested in is the
    GRT on the focal plane, its serial number (29062) and channel (1) have been set as
    default values.
    """
    Rmeas = 0
    while Rmeas == 0:
        try:
            Rmeas = getRes(GRT_chan)
        except pyvisa.VisaIOError:
            print("VisaIOError... SIM921. Let's try again!")
            time.sleep(1)
    return R2T_interp(Rmeas, GRT_serial)


def setRange(GRT_chan=1, Range=5):
    """
    Sets the range of the SIM921 to read the GRT 
    Mostly useful when overloaded or stuck
    Useful range is usually 6 (20kOhms)
    To unstuck the readout, just set the range to 5, then 6 again.
    """
    rm = pyvisa.ResourceManager()    
    inst = rm.open_resource('GPIB0::%d::INSTR'%num_gpib)
    inst.read_termination = '\n'
    inst.write_termination = '\n'
    inst.write('xyzzy')
    # time.sleep(0.01)
    inst.write('CONN %d,"xyzzy"'%GRT_chan)
    # time.sleep(0.01)
    inst.write('RANG %d'%Range)
    inst.write('RANG?')
    newrange = inst.read()
    inst.close()
    rm.close()
    return newrange

# def ReleaseTsensor():
#     rm = pyvisa.ResourceManager()
#     inst = rm.open_resource('GPIB0::%d::INSTR'%num_gpib)
#     inst.read_termination = '\n'
#     inst.write_termination = '\n'
#     inst.write('xyzzy')
#     inst.write('xyzzy')
  
    
#for i in range(100):
#    print(GetTemp())
#  
# def R2T(r):
#     """
#     Calculate the temperature from resistance
#     Input:resistance [Ohm]
#     Output:temperature [K]
#     """
#     a = np.asarray([0.472833, -0.542306,
#                     0.190448, -0.050404,
#                     0.023272, -0.015937,
#                     0.005533, 0.000199,
#                     0.0, 0.0,
#                     0.0, 0.0])
#     zl = 1.88165507088
#     zu = 4.09620414545
#     if r < 78:
#         #raise ValueError("Temperature is too high!")
#         print("Focal plane temperature is higher than 1.27 K, cannot return an accurate temperature.")
#         return 6
#     z = np.log10(r)
#     x = ((z - zl) - (zu - z))/(zu - zl)
#     i = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
#     b = i*math.acos(x)
#     t = np.dot(a,np.cos(b))
#     return t
    

# def T2R(temp):
#     """
#     Calculate the resistance of the temperature
#     Input:temperature [K]
#     Output:resistance [Ohm]
#     """
#     tolerance = 0.00001
#     delta = 1.0
#     Res = 500.
    
#     while delta>tolerance:
#         temp_T = R2T(Res)
#         delta = temp_T/temp-1.
#         Res = Res*(delta+1)
        
#     return Res  
    


        
        
        


