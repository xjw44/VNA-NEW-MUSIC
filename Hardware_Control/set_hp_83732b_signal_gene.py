# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:43:47 2021

@author: kids
"""
import pyvisa
import sys

rm = pyvisa.ResourceManager()
sig_gen = rm.open_resource("GPIB::19::INSTR")

def read_parameters():
    print('frequency set to %.6f MHz'%(float(sig_gen.query('freq?'))*1e-6))
    print('power set to %.2f dBm'%(float(sig_gen.query('pow?'))))
    print('pulsing period is %.3f ms'%(float(sig_gen.query('pulse:period?'))*1e3))
    print('pulse width is %.3f us'%(float(sig_gen.query('pulse:width?'))*1e6))
    out_pulse = sig_gen.query('PULM:STATe?')
    if out_pulse == '+0\n':
        pulse_state_str = 'OFF'
    elif out_pulse == '+1\n':
        pulse_state_str = 'ON'
    print('pulse state is ' +  pulse_state_str)
    outp = sig_gen.query('outp?')
    if outp == '+0\n':
        display_str = 'OFF'
    elif outp == '+1\n':
        display_str = 'ON'
    print('signal generator is ' +  display_str)

def on_off_pulse(onoroff):
    if onoroff == 0:
        sig_gen.write('PULM:STATe 0')
        output = sig_gen.query('PULM:STATe?')
    elif onoroff == 1:
        sig_gen.write('PULM:STATe 1')
        output = sig_gen.query('PULM:STATe?')
    print('signal generator pulse set to ' + output)
    
def on_off_switch(onoroff):
    if onoroff == 0:
        sig_gen.write('outp 0')
        output = sig_gen.query('outp?')
    elif onoroff == 1:
        sig_gen.write('outp 1')
        output = sig_gen.query('outp?')
    print('signal generator output set to ' + output)

def set_power(power):
    print('previous power was %.1f dBm'%(float(sig_gen.query('pow?'))))
    sig_gen.write('power ' + str(power))
    print('new power set to %.1f dBm'%(float(sig_gen.query('pow?'))))

def set_frequency(frequency):
    print('previous frequency was %.3f MHz'%(float(sig_gen.query('freq?'))*1e-6))
    sig_gen.write('frequency ' + str(frequency))
    print('new frequency set to %.3f MHz'%(float(sig_gen.query('freq?'))*1e-6))

def set_pulse_params(width, period):
    print('previous period and width were: %.2e s and %.2e s'%(float(sig_gen.query('pulse:period?')), float(sig_gen.query('pulse:width?'))))
    sig_gen.write('pulse:width %.9f'%(width))
    sig_gen.write('pulse:period %.9f'%(period))
    print('new period and width are: %.2e s and %.2e s'%(float(sig_gen.query('pulse:period?')), float(sig_gen.query('pulse:width?'))))
    sig_gen.write('pm:state 0')
    print(sig_gen.query('pm:state?'))
    

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