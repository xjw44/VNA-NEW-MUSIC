# -*- coding: utf-8 -*-
"""
Created on Mar 26 2020

@author: F Defrance
"""

import pyvisa
import time

#rm = pyvisa.ResourceManager()    
#inst = rm.open_resource('GPIB0::15::INSTR')
#
#def set2zero(out, verb=1):
#    print("Output %d set to 0V and turned off"%(out))
#    inst.write('inst:sel OUT%d'%out)
#    inst.write('sour:volt 0')
#    inst.write('OUTPut:STATe 0')
#       
#def initialize(out, Vmax = 10, verb=1):  
#    if verb: print("--- Initialize output %d ---"%(out))
#    set2zero(out, verb=verb)
#    inst.write('SOURce:CURRent:PROTect:STATe ON')
#    inst.write('VOLTage:PROTection %.1f'%Vmax)
#    if verb: print("Max voltage set to %.1f V"%(Vmax))
#    inst.write('sour:curr 1')
#        
#def writeV(out, setV, verb=1):
#    if verb: print("--- Set output %d to %.3f V ---"%(out, setV))
#    inst.write('inst:sel OUT%d'%out)
#    inst.write('sour:volt %.3f'%setV)
#    inst.write('OUTPut:STATe 1')
#    
#def readIV(out, verb=1):
#    if verb: print("--- Read state of output %d ---"%out)
#    inst.write('inst:sel OUT%d'%out)
#    inst.write('meas:VOLT?')
#    getV = float(inst.read())
#    inst.write('meas:CURR?')
#    getI = float(inst.read())
#    inst.write('OUTPut:STAT?')
#    state = float(inst.read()) 
#    if verb: print("Output State: %d \nOutput voltage: %.3f V \nOutput current: %.4f A "%(state, getV, getI))
#    return getV, getI
    


class WrongInstrumentError(Exception):
    """The wrong instrument is connected

    A connection was successfuly established, and the instrument responded
    to a request to identify itself, but the ID recieved was wrong.
    Probably the instrument at the given VISA identifier is not the one
    you wanted.    
    """
    pass
        
class PS2520G:
    def __init__(self, out, verb=0):
        self.out = out
        self.verb = verb
        self.Vmax = 10
        self.rm = None
        self.inst = None
        
    def __enter__(self):
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource('GPIB0::15::INSTR')
        return self

    def set2zero(self):
        print("Output %d set to 0V and turned off"%(self.out))
        self.inst.write('inst:sel OUT%d'%self.out)
        self.inst.write('sour:volt 0')
        self.inst.write('OUTPut:STATe 0')
           
    def initialize(self):  
        if self.verb: print("--- Initialize output %d ---"%(self.out))
        self.set2zero()
        self.inst.write('SOURce:CURRent:PROTect:STATe ON')
        self.inst.write('VOLTage:PROTection %.1f'%self.Vmax)
        if self.verb: print("Max voltage set to %.1f V"%(self.Vmax))
        self.inst.write('sour:curr 1')
            
    def writeV(self, setV):
        if self.verb: print("--- Set output %d to %.3f V ---"%(self.out, setV))
        self.inst.write('inst:sel OUT%d'%self.out)
        self.inst.write('sour:volt %.3f'%setV)
        self.inst.write('OUTPut:STATe 1')
        
    def readIV(self):
        if self.verb: print("--- Read state of output %d ---"%self.out)
        self.inst.write('inst:sel OUT%d'%self.out)
        self.inst.write('meas:VOLT?')
        getV = float(self.inst.read())
        self.inst.write('meas:CURR?')
        getI = float(self.inst.read())
        self.inst.write('OUTPut:STAT?')
        state = float(self.inst.read()) 
        if self.verb: print("Output State: %d \nOutput voltage: %.3f V \nOutput current: %.4f A "%(state, getV, getI))
        return getV, getI       
    
    def __exit__(self, exc_type, exc_val, exc_tb):
#        print('__exit__ called')
#        flag_exit = 0
#        while flag_exit == 0:
        try:
            #self.inst.close()
            self.rm.close()       
#            flag_exit = 1
#            print("closed ok")
        except pyvisa.VisaIOError:
            print("VisaIOError... Probably just a collision (PS2520G). Let's try again!")
            time.sleep(1)

#
#for i in range(20):
# with PS2520G(1, verb=1) as ps:
#     ps.initialize()
#     time.sleep(1)
#     ps.readIV()
#     ps.writeV(0.1)
#     time.sleep(1)
#     ps.readIV()
#        
#    get_diode_temp(5, verb=1)
#    SetSMASwitch(1)
#    getRes(1)
#    time.sleep(0.2)
#    SetSMASwitch(2)
#    time.sleep(0.2)
#    
#    
    








    
    