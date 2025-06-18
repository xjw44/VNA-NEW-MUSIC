# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 14:15:01 2021

@author: kids
In order to use the NI USB to RS 232 hub with Windows 10, you need to edit the registery by importing serial.reg file available on NI website:
https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z0000019Q4fSAE&l=en-US

User manual of the SR560 with commands is here: https://www.thinksrs.com/downloads/pdfs/manuals/SR560m.pdf

CPLGi : Input coupling (i=1:DC, i=2:AC)
FLTM i:  Sets filter mode. (i = 
0 = bypass,  
1 = 6 dB low pass, 
2 = 12 dB low pass,  
3 = 6 dB high pass,  
4 = 12 dB highpass,  
5 = bandpass)
GAIN i Sets the gain. 
i = 0 – 14 = 1, 2, 5, ... 50 k gain
HFRQi Sets highpass filter frequency. 
I = 0 – 11 sets frequency = 0.03 Hz to 10 kHz
LALL Listen all. Makes all attached SR560’s listeners.
LFRQ i Sets lowpass filter frequency. 
i = 0 – 15 sets frequency = 0.03 Hz to 1 MHz
[0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1k, 3k, 10k, 30k, 100k, 300k, 1M]
UNLS Unlisten. Unaddresses all attached SR560’s
*RST Reset. Recalls default settings.

When the coupling is set to AC, a
0.03 Hz cutoff high-pass filter is always
engaged. All high-pass filter modes can still
be selected while AC-coupled, but the 0.03
Hz filter will always be in, even if the filters
are set to DC. Because one of the two filter
sections is always used as a high pass
when AC coupling is selected, low-pass
filters are only available with a 6 dB / octave
rolloff.

"""

import pyvisa
import numpy as np
from pyvisa.constants import StopBits, Parity

def init_connection():
    rm = pyvisa.ResourceManager()
    rm.list_resources()
    
    inst1 = rm.open_resource('ASRL3::INSTR', 
        baud_rate=9600, data_bits=8, flow_control=4,
        parity=Parity.none, stop_bits=StopBits.two)
    inst1.write_termination = '\r\n'
    
    inst2 = rm.open_resource('ASRL4::INSTR', 
        baud_rate=9600, data_bits=8, flow_control=4,
        parity=Parity.none, stop_bits=StopBits.two)
    inst2.write_termination = '\r\n'
    return inst1, inst2


def set_DC_mode(gain=100):
    inst1, inst2 = init_connection()
    for inst in [inst1, inst2]:
        inst.write('LALL')
        inst.write('CPLG1')
        # inst.write('FLTM 1')
        # inst.write('FLTM 2')
        # inst.write('LFRQ14')
        # inst.write('LFRQ10')
        inst.write('UNLS')
    set_gain(gain)
        

def set_AC_mode(gain=200):
    inst1, inst2 = init_connection()
    for inst in [inst1, inst2]:
        inst.write('LALL')
        inst.write('CPLG2')
        # inst.write('FLTM 5')
        # inst.write('LFRQ14')
        # inst.write('LFRQ11')
        # inst.write('HFRQ2')      
        # inst.write('HFRQ1')      
        inst.write('UNLS')
    set_gain(gain)
        
        
def set_gain(gain):
    inst1, inst2 = init_connection()
    gain_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    if int(gain) in gain_list:
        gain_i = gain_list.index(gain)
        for inst in [inst1, inst2]:
            inst.write('LALL')
            inst.write('GAIN %d'%gain_i)
            inst.write('UNLS')
        print('SR560 gain set to %d'%gain_list[gain_i])
    else:
        print('Error! Requested gain not valid. Nothing changed')
        return
    
    
def set_AC_LPfilter(LP=10e3):
    # Set low pass filter freq, for AC mode
    # High pass filter is always set to 0.03 Hz
    inst1, inst2 = init_connection()
    LP_list = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5, 1e6]
    if not int(LP) in LP_list:
        print('Error! Requested low pass cutoff frequency (%.1e Hz) not valid.'%LP)
        LP = LP_list[np.argmin(np.abs(LP - np.array(LP_list)))]
        print('Low pass cutoff frequency set to %1.e Hz'%LP)
    LP_i = LP_list.index(LP)
    for inst in [inst1, inst2]:
        inst.write('LALL')
        inst.write('CPLG2') # AC
        inst.write('FLTM 5') # bandpass mode
        inst.write('LFRQ%d'%LP_i)
        inst.write('UNLS')
    print('SR560 preamps: AC mode')
    print('HP cutoff = 0.03 Hz (fixed), LP cutoff = %1.e Hz'%LP)


def set_DC_filters(LP=None, HP=None):
    # Set low  and high pass filters freq, for DC mode
    inst1, inst2 = init_connection()
    HP_list = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4]
    LP_list = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5, 1e6]
    if LP == None and HP == None:
        print('SR560 preamps: No filter mode')
        FLTM = 0
        return
    elif HP == None:
        print('SR560 preamps: Low Pass mode requested with 12dB/oct')
        FLTM = 2
        if not LP in LP_list:
            print('Error! Requested low pass cutoff frequency (%.3f Hz) not valid.'%LP)
            LP = LP_list[np.argmin(np.abs(LP - np.array(LP_list)))]
            print('Low pass cutoff frequency set to %3.f Hz'%LP)              
        LP_i = LP_list.index(LP)
    elif LP == None:
        print('SR560 preamps: High Pass mode requested with 12dB/oct')
        FLTM = 4
        if not HP in HP_list:
            print('Error! Requested high pass cutoff frequency (%.3f Hz) not valid.'%HP)
            HP = HP_list[np.argmin(np.abs(HP - np.array(HP_list)))]
            print('High pass cutoff frequency set to %3.f Hz'%HP) 
        HP_i = HP_list.index(HP)
    else:
        print('SR560 preamps: Band Pass mode requested with 6dB/oct for each filter')
        FLTM = 5
        if not LP in LP_list:
            print('Error! Requested low pass cutoff frequency (%.3f Hz) not valid.'%LP)
            LP = LP_list[np.argmin(np.abs(LP - np.array(LP_list)))]
            print('Low pass cutoff frequency set to %3.f Hz'%LP)
        if not HP in HP_list:
            print('Error! Requested high pass cutoff frequency (%.3f Hz) not valid.'%HP)
            HP = HP_list[np.argmin(np.abs(HP - np.array(HP_list)))]
            print('High pass cutoff frequency set to %3.f Hz'%HP)    
        LP_i = LP_list.index(LP)
        HP_i = HP_list.index(HP)
  
    for inst in [inst1, inst2]:
        inst.write('LALL')
        inst.write('CPLG1') # DC
        inst.write('FLTM %d'%FLTM) # filter mode, LP, HP, or BP
        if not LP == None:
            inst.write('LFRQ%d'%LP_i)
            print('SR560 preamps: DC mode, LP cutoff frequency set to %.1e Hz'%LP)
        if not HP == None:
            inst.write('HFRQ%d'%HP_i)
            print('SR560 preamps: DC mode, HP cutoff frequency set to %.3f Hz'%HP)
        inst.write('UNLS')

        
        
