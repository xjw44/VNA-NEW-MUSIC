# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:43:47 2021

@author: kids
"""
import pyvisa
import sys

rm = pyvisa.ResourceManager()
sig_gen = rm.open_resource("GPIB::17::INSTR")

def read_parameters():
    print('frequency set to %.6f MHz'%(float(sig_gen.query('OF1'))))
    print('power set to %.2f dBm'%(float(sig_gen.query('OL1'))))
    
def on_off_switch(onoroff):
    if onoroff == 0:
        sig_gen.write('RF0')
        output = 'OFF'
    elif onoroff == 1:
        sig_gen.write('RF1')
        output = 'ON'
    print('signal generator output set to ' + output)

def set_power(power):
    print('previous power was %.2f dBm'%(float(sig_gen.query('OL1'))))
    sig_gen.write('L1 ' + str(power) + ' DM')
    print('new power set to %.1f dBm'%(float(sig_gen.query('OL1'))))

def set_frequency(frequency):
    print('previous frequency was %.3f MHz'%(float(sig_gen.query('OF1'))))
    sig_gen.write('F1 ' + str(frequency) + ' MH')
    print('new frequency set to %.3f MHz'%(float(sig_gen.query('OF1'))))


    

# if __name__ == "__main__":
#     if len(sys.argv) == 1:
#         read_parameters()
#     if len(sys.argv) >= 2:
#         onoroff = int(sys.argv[1])
#         on_off_switch(onoroff)
#     if len(sys.argv) >= 3:
#         power = float(sys.argv[2])
#         set_power(power)
#     if len(sys.argv) >= 4:
#         frequency = float(sys.argv[3])
#         set_frequency(frequency)