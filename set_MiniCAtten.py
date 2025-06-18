# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:14:10 2023

@author: kids Shiling Yu

Set new attenuator that replace broken weinschel
"""

import numpy as np
import clr # pythonnet
clr.AddReference('Hardware_Control/mcl_RUDAT_NET45')      # Reference the DLL

from mcl_RUDAT_NET45 import USB_RUDAT

def setAtten(attn):
    """
    Sets the attenuation of the Minicircuits variable attenuator on each of the input and output dewar
    Model No. RCDAT-6000-90
    Control applications from 1 to 6000 MHz
    The attenuator provides 0 to 90 dB attenuation in 0.25 dB steps
    Negative numbers are also allowed and will take the sbsolute value
    """
    att1 = USB_RUDAT()   # Create an instance of the USB control class
    att2 = USB_RUDAT()   # Create an instance of the USB control class
    
    Status1 = att1.Connect("12208030133")       # Connect (pass the serial number as an argument if required)
    Status2 = att2.Connect("12208030135")       # Connect (pass the serial number as an argument if required)
    Arange = (0, 90)
    attn=abs(attn)
    
    if Status1[0] > 0:              # The connection was successful
    
        # Responses = att.Send_SCPI(":SN?", "")        # Read serial number
        # print (str(Responses[1]))   # Python interprets the response as a tuple [function return (0 or 1), command parameter, response parameter]
    
        # Responses = att.Send_SCPI(":MN?", "")        # Read model name
        # print (str(Responses[1]))
        
        # Set and then read attenuation (single channel models)
        if (np.asarray(attn[0]) >= Arange[0]).all() and (np.asarray(attn[0]) <= Arange[1]).all():
            if attn[0]%(0.25) != 0:
                attn[0]=round(attn[0] / 0.25) * 0.25
                print("!the attenuation need to be 0.25 dB steps")
            Status = att1.Send_SCPI(":SETATT="+str(attn[0]),"")    # Set attenuation
            Responses = att1.Send_SCPI(":ATT?", "")   # Read attenuation
            print('Attenuator channel 1 set to %.2f dB'%(float(Responses[1])))
            att1.Disconnect()             # Disconnect at the end of the program
        else:
            print("Wrong attenuator1 setting value range")
            return
    
    else:
        print ("Could not connect channel 1 '12208030133'.") #correspond to in to the cryostat
    
    if Status2[0] > 0:              # The connection was successful
    
        # Set and then read attenuation (single channel models)
        if (np.asarray(attn[1]) >= Arange[0]).all() and (np.asarray(attn[1]) <= Arange[1]).all():
            if attn[1]%(0.25) != 0:
                attn[1]=round(attn[1] / 0.25) * 0.25
                print("!the attenuation need to be 0.25 dB steps")
            Status = att2.Send_SCPI(":SETATT="+str(attn[1]), "")    # Set attenuation
            Responses = att2.Send_SCPI(":ATT?", "")   # Read attenuation
            print('Attenuator channel 2 set to %.2f dB'%(float(Responses[1])))
            att2.Disconnect()             # Disconnect at the end of the program
        else:
            print("Wrong attenuator2 setting value range")
            return
    
    else:
        print ("Could not connect channel 2 '12208030135'.") #correspond to out from cryostat


if __name__ == "__main__":
    # setAtten(np.array([38,10])) #historically Weinschel was set channel 1 at 38 and channel 2 at 10
    # setAtten(np.array([20,10])) 
    setAtten(np.array([10,10])) #Be231102d2 for USRP
    

