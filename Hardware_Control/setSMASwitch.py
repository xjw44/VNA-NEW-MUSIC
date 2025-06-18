# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:23:50 2019

@author: F Defrance

Function to switch the 2 SMA switches (at input and output of the Yellow cryostat)
in order to select one of the 4 channels (and devices).
Initially I had to call the Matlab function to do the switch since Minicircuit did not provide any way to do it with Python 64 bits.
see: https://www.minicircuits.com/softwaredownload/Prog_Examples_Troubleshooting.pdf
Recently they provided us a dll library that allows the switch control from python.
So the function calling Matlab is commented and the new function using the dll is activated.
"""

# import matlab.engine

# def SetSMASwitch(port):
#     eng = matlab.engine.start_matlab()
#     eng.cd(r'C:\Users\kids\Documents\MatlabCode', nargout=0)
#     status = eng.SetSMASwitch(float(port))
#     eng.quit()
#     if status != 0:
#         print('SMA switches set to port %d'%status)
#     else:
#         print('Error: SMA switches not set')


# SetSMASwitch(4)
        
import clr # pythonnet
clr.AddReference('Hardware_Control\mcl_SolidStateSwitch64') 	# Reference the DLL
clr.AddReference('Hardware_Control\mcl_SolidStateSwitch_NET45')
from mcl_SolidStateSwitch64 import USB_Digital_Switch
from mcl_SolidStateSwitch_NET45 import USB_Digital_Switch

import time

def SetSMASwitch(port):
#    print('start SMAswitch')
    time.sleep(0.5)
    sw1 = USB_Digital_Switch()   # Create an instance of the switch class
    sw2 = USB_Digital_Switch()   # Create an instance of the switch class

    connect = 0
    while connect == 0:
        sw1.Connect('11612250003')       # Connect the switch (pass the serial number as an argument if required)
        sw2.Connect('12201310079')       # Connect the switch (pass the serial number as an argument if required)
        time.sleep(0.5)
        sw1.Set_SP4T_COM_To(port)      # Set switch to port 
        sw2.Set_SP4T_COM_To(port)      # Set switch to port 
        if sw1.Get_SP4T_State() == port and sw2.Get_SP4T_State() == port:
            print('SMA switches set to port %d'%port)
            connect = 1
        else:
            print('Error: SMA switches not set. try again')    
            time.sleep(1)
        sw1.Disconnect()            # Disconnect at the end of the program
        sw2.Disconnect()            # Disconnect at the end of the program

#import win32com.client # Allows communication via COM interface
#
##Instantiate COM client
#app = win32com.client.Dispatch("TRVNA.application")
#time.sleep(1)
#
#def setup_VNA(Ch, Tr):
#    app.scpi.system.preset()
#    time.sleep(0.2)
#    app.scpi.GetCALCulate(Ch).GetPARameter(Tr).define = 'S21'
#    app.scpi.GetCALCulate(Ch).GetPARameter(Tr).select()
#    # Set the computer signal as trigger source 
#    # (required to have the VNA wait until the end of the measurement before accepting other commands)
#    app.scpi.TRIGger.SEQuence.Source = "BUS" 
#    # disable display
#    app.scpi.display.enable = False
#    
#    
#    
#    
#
#import time
#for i in range(10):
#    SetSMASwitch(1)
#    setup_VNA(1, 1)
##    time.sleep(0.5)
#    SetSMASwitch(2)
#    setup_VNA(1, 1)
##    time.sleep(0.5)
    # SetSMASwitch(3)
    # setup_VNA(1, 1)
##    time.sleep(0.5)
    # SetSMASwitch(4)
#    setup_VNA(1, 1)
##    time.sleep(0.5)

if __name__ == "__main__":
    SetSMASwitch(4)
