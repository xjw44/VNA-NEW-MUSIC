# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:26:03 2024

@author: Shiling Yu

Box 4 Port to: 1:Mixer 2:Mixer(High-freq) 3:USRP 4:VNA
"""

#
import clr # pythonnet
# clr.AddReference('mcl_SolidStateSwitch64') 	# Reference the DLL
clr.AddReference('mcl_SolidStateSwitch_NET45')
# from mcl_SolidStateSwitch64 import USB_Digital_Switch
from mcl_SolidStateSwitch_NET45 import USB_Digital_Switch

import time

def SetBoxSwitch(port):
    '''this is the help
        Port:
        1: LF IQ mixer box
        2: HF IQ mixer box
        3: USRP
        4: VNA
        only 4 ports
        '''
#    print('start SMAswitch')
    time.sleep(0.5)
    sw1 = USB_Digital_Switch()   # Create an instance of the switch class
    sw2 = USB_Digital_Switch()   # Create an instance of the switch class
    
    if port==1 or port==2 or port==3 or port==4:
        connect = 0
        while connect == 0:
            sw1.Connect('12301170102')       # Connect the switch (pass the serial number as an argument if required)
            sw2.Connect('12301170107')       # Connect the switch (pass the serial number as an argument if required)
            time.sleep(0.5)
            sw1.Set_SP4T_COM_To(port)      # Set switch to port 
            sw2.Set_SP4T_COM_To(port)      # Set switch to port 
            if sw1.Get_SP4T_State() == port and sw2.Get_SP4T_State() == port:
                print('BoxPort switches set to port %d'%port)
                connect = 1
            else:
                print('Error: BoxPort switches not set. try again')    
                time.sleep(1)
            sw1.Disconnect()            # Disconnect at the end of the program
            sw2.Disconnect()            # Disconnect at the end of the program
    else:
        print("Wrong port setting value (only 1 2 3 4)")
    
if __name__ == "__main__":
    SetBoxSwitch(4) #VNA
    # SetBoxSwitch(3) #USRP
     #SetBoxSwitch(1) #IQ Mixer
