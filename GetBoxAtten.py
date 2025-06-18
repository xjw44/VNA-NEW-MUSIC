# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:02:51 2023

@author: Shiling

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pylab as pl


plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.size']=20


### Data fitting process
# root_folder = 'D:/data/TransmissionLine/'
# att=50
# baseline_data = np.loadtxt(open(root_folder + 'InOutput_baseline1.csv','rb'), skiprows=3, delimiter=',',usecols=[0,1])
# Port1in = np.loadtxt(open(root_folder + 'Port1in_att'+str(att)+'.csv','rb'), skiprows=3, delimiter=',',usecols=[0,1])
# Port1out = np.loadtxt(open(root_folder + 'Output_port1_att'+str(att)+'.csv','rb'), skiprows=3, delimiter=',',usecols=[0,1])

# f_range_board1 = (10e6, 4000e6)
# freq = baseline_data[:,0]

# S21_Port1in_output = Port1in[:,1] - baseline_data[:,1]
# S21_Port1out_input = Port1out[:,1] - baseline_data[:,1]

# i_start1 = np.argmin(np.abs(f_range_board1[0] - freq))
# i_stop1 = np.argmin(np.abs(f_range_board1[1] - freq))

# order = 4
# z1 = np.polyfit(freq[i_start1:i_stop1], S21_Port1in_output[i_start1:i_stop1], order)
# p1 = np.poly1d(z1)
# print(p1)
# z2 = np.polyfit(freq[i_start1:i_stop1], S21_Port1out_input[i_start1:i_stop1], order)
# p2 = np.poly1d(z2)
# print(p2)

# plt.figure()
# plt.plot(freq[i_start1:i_stop1]/1e6, S21_Port1in_output[i_start1:i_stop1], label="data: Port1in_output att{}".format(att))
# plt.plot(freq[i_start1:i_stop1]/1e6, p1(freq[i_start1:i_stop1]), label="fit: Port1in-output att{}".format(att))
# plt.ylabel('|S21|  [dB]')
# plt.xlabel('Frequency  [MHz]')
# plt.grid()
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(freq[i_start1:i_stop1]/1e6, S21_Port1out_input[i_start1:i_stop1], label="data: Port1out_input att{}".format(att))
# plt.plot(freq[i_start1:i_stop1]/1e6, p2(freq[i_start1:i_stop1]), label="fit: Port1out_input att{}".format(att))
# plt.ylabel('|S21|  [dB]')
# plt.xlabel('Frequency  [MHz]')
# plt.grid()
# plt.legend()
# plt.show()


### Fitting results
'''
att0
Port1in_output
           5             4             3             2
6.949e-48 x - 3.398e-38 x - 1.221e-28 x + 1.007e-18 x - 3.145e-09 x - 3.492
Port1out_input
           5             4             3             2
9.309e-48 x - 4.854e-38 x - 1.016e-28 x + 9.753e-19 x - 2.955e-09 x - 3.027

att5
           5             4            3             2
7.401e-48 x - 5.049e-38 x + 6.16e-30 x + 6.117e-19 x - 2.66e-09 x - 8.409
           5             4             3             2
1.201e-47 x - 9.164e-38 x + 1.481e-28 x + 3.273e-19 x - 2.246e-09 x - 8.263

att10
           5             4             3             2
4.596e-48 x - 3.116e-38 x + 1.275e-32 x + 4.235e-19 x - 2.297e-09 x - 13.53
           5             4             3             2
9.046e-48 x - 7.187e-38 x + 1.456e-28 x + 1.105e-19 x - 1.814e-09 x - 13.23

att20
           5             4             3             2
7.583e-48 x - 6.618e-38 x + 1.514e-28 x + 1.114e-19 x - 2.002e-09 x - 23.34
           5             4             3             2
1.205e-47 x - 1.093e-37 x + 3.221e-28 x - 2.924e-19 x - 1.395e-09 x - 23.05

att30
           5            4             3             2
7.109e-48 x - 6.44e-38 x + 1.634e-28 x + 2.896e-20 x - 1.893e-09 x - 33.37
           5            4             3             2
1.074e-47 x - 9.81e-38 x + 2.959e-28 x - 3.121e-19 x - 1.314e-09 x - 33.04

att40
           5             4            3             2
5.089e-48 x - 4.896e-38 x + 1.33e-28 x + 1.778e-20 x - 1.87e-09 x - 43.4
           5             4            3             2
1.076e-47 x - 1.013e-37 x + 3.23e-28 x - 3.935e-19 x - 1.253e-09 x - 42.79

att50
            5             4             3             2
-1.469e-47 x + 1.425e-37 x - 4.933e-28 x + 7.719e-19 x - 2.061e-09 x - 52.97
            5             4             3             2
-1.559e-47 x + 1.509e-37 x - 5.069e-28 x + 6.694e-19 x - 1.667e-09 x - 52.68

att60
            5             4             3             2
-3.499e-48 x + 3.584e-38 x - 1.409e-28 x + 2.867e-19 x - 1.84e-09 x - 63.03
            5             4             3             2
-1.297e-47 x + 1.226e-37 x - 3.955e-28 x + 4.679e-19 x - 1.55e-09 x - 62.65

att70
          5             4             3            2
9.62e-48 x - 7.869e-38 x + 1.917e-28 x - 1.52e-19 x - 1.473e-09 x - 72.99
            5             4             3             2
-2.374e-47 x + 2.185e-37 x - 7.341e-28 x + 1.045e-18 x - 1.93e-09 x - 72.49

att80
            5             4             3             2
-1.302e-47 x + 1.327e-37 x - 5.175e-28 x + 9.993e-19 x - 1.926e-09 x - 82.67
            5             4             3             2
-8.088e-48 x + 1.307e-38 x + 2.187e-28 x - 7.478e-19 x - 2.854e-10 x - 82.69

att90
            5             4            3             2
-1.381e-47 x + 1.149e-37 x - 3.71e-28 x + 6.333e-19 x - 5.877e-10 x - 87.98
            5            4             3             2
-1.305e-46 x + 1.28e-36 x - 4.549e-27 x + 7.167e-18 x - 4.924e-09 x - 86.81
'''

def get_atten_SwitchBox(freq, atten_in, atten_out):
    '''
    freq in Hz
    This is the total attenuation of the four ports of the reading switch box: 
        from the port input to the switch to the attenuator to the box output, 
        plus from the Dewar input to the attenuator to the switch to the port output
    Reduced measured cable baseline.
    From 10 MHz to 4000 MHz
    The result for four different ports below 3GHz are almost identical,
        except for the 4th port which has a slight deviation near 4GHz.
    The attenuation is related to the attenuator setting value. (between 0 and 90)
    
    atten_in -> from the port input to the switch to the attenuator to the box output to Dewar
    atten_out -> from the Dewar input to the attenuator to the switch to the port output
        The attenuator provides 0 to 90 dB attenuation in 0.25 dB steps
        Negative numbers are also allowed and will take the sbsolute value
    
    Measurement from 12/15/2023
    '''
    attenX=[0,5,10,20,30,40,50,60,70,80,90]
    pi1 = np.array([ 6.949e-48, -3.398e-38, -1.221e-28, 1.007e-18, -3.145e-09, -3.492 ])
    po1 = np.array([ 9.309e-48, -4.854e-38, -1.016e-28, 9.753e-19, -2.955e-09, -3.027 ])
    pi2 = np.array([ 7.401e-48, -5.049e-38, 6.160e-30, 6.117e-19, -2.660e-09, -8.409 ])
    po2 = np.array([ 1.201e-47, -9.164e-38, 1.481e-28, 3.273e-19, -2.246e-09, -8.263 ])
    pi3 = np.array([ 4.596e-48, -3.116e-38, 1.275e-32, 4.235e-19, -2.297e-09, -13.53 ])
    po3 = np.array([ 9.046e-48, -7.187e-38, 1.456e-28, 1.105e-19, -1.814e-09, -13.23 ])
    pi4 = np.array([ 7.583e-48, -6.618e-38, 1.514e-28, 1.114e-19, -2.002e-09, -23.34 ])
    po4 = np.array([ 1.205e-47, -1.093e-37, 3.221e-28, -2.924e-19, -1.395e-09, -23.05 ])
    pi5 = np.array([ 7.109e-48, -6.44e-38, 1.634e-28, 2.896e-20, - 1.893e-09, - 33.37 ])
    po5 = np.array([ 1.074e-47, -9.81e-38, 2.959e-28, - 3.121e-19, - 1.314e-09, - 33.04 ])
    pi6 = np.array([ 5.089e-48, -4.896e-38, 1.33e-28, 1.778e-20, - 1.87e-09, - 43.4 ])
    po6 = np.array([ 1.076e-47, -1.013e-37, 3.23e-28, - 3.935e-19, - 1.253e-09, - 42.79 ])
    pi7 = np.array([ -1.469e-47, 1.425e-37, - 4.933e-28, 7.719e-19, - 2.061e-09, - 52.97 ])
    po7 = np.array([ -1.559e-47, 1.509e-37, - 5.069e-28, 6.694e-19, - 1.667e-09, - 52.68 ])
    pi8 = np.array([ -3.499e-48, 3.584e-38, - 1.409e-28, 2.867e-19, - 1.84e-09, - 63.03 ])
    po8 = np.array([ -1.297e-47, 1.226e-37, - 3.955e-28, 4.679e-19, - 1.55e-09, - 62.65 ])
    pi9 = np.array([ 9.62e-48, - 7.869e-38, 1.917e-28, - 1.52e-19, - 1.473e-09, - 72.99 ])
    po9 = np.array([ -2.374e-47, 2.185e-37, - 7.341e-28, 1.045e-18, - 1.93e-09, - 72.49 ])
    pi10 = np.array([ -1.302e-47, 1.327e-37, - 5.175e-28, 9.993e-19, - 1.926e-09, - 82.67 ])
    po10 = np.array([ -8.088e-48, 1.307e-38, 2.187e-28, - 7.478e-19, - 2.854e-10, - 82.69 ])
    pi11 = np.array([ -1.381e-47, 1.149e-37, - 3.71e-28, 6.333e-19, - 5.877e-10, - 87.98 ])
    po11 = np.array([ -1.305e-46, 1.28e-36, - 4.549e-27, 7.167e-18, - 4.924e-09, - 86.81 ])
    
    
    pi = np.array([pi1,pi2,pi3,pi4,pi5,pi6,pi7,pi8,pi9,pi10,pi11])
    po = np.array([po1,po2,po3,po4,po5,po6,po7,po8,po9,po10,po11])
    f11=interpolate.interp1d(attenX, pi[:,0],kind='linear')
    f12=interpolate.interp1d(attenX, pi[:,1],kind='linear')
    f13=interpolate.interp1d(attenX, pi[:,2],kind='linear')
    f14=interpolate.interp1d(attenX, pi[:,3],kind='linear')
    f15=interpolate.interp1d(attenX, pi[:,4],kind='linear')
    f16=interpolate.interp1d(attenX, pi[:,5],kind='linear')
    f21=interpolate.interp1d(attenX, po[:,0],kind='linear')
    f22=interpolate.interp1d(attenX, po[:,1],kind='linear')
    f23=interpolate.interp1d(attenX, po[:,2],kind='linear')
    f24=interpolate.interp1d(attenX, po[:,3],kind='linear')
    f25=interpolate.interp1d(attenX, po[:,4],kind='linear')
    f26=interpolate.interp1d(attenX, po[:,5],kind='linear')
    # attenXN=np.linspace(0,50,100)
    # piA=f1(attenXN)
    # plt.figure()
    # plt.plot(attenXN, piA)
    # plt.grid()
    # plt.show()
    Frange = (10e6, 3000e6)
    Arange = (0, 90)
    atten_in=abs(atten_in)
    atten_out=abs(atten_out)
    
    if (np.asarray(freq) >= Frange[0]).all() and (np.asarray(freq) <= Frange[1]).all():
        if (np.asarray(atten_in) >= Arange[0]).all() and (np.asarray(atten_in) <= Arange[1]).all():
            if atten_in%(0.25) != 0:
                atten_in=round(atten_in/0.25)*0.25 
                print("the 1st input attenuation need to be 0.25 dB steps")
            if atten_out%(0.25) != 0:
                atten_out=round(atten_out/0.25)*0.25
                print("the 2nd output attenuation need to be 0.25 dB steps")
            pif = np.poly1d([ f11(atten_in), f12(atten_in), f13(atten_in), f14(atten_in), f15(atten_in), f16(atten_in) ])
            pof = np.poly1d([ f21(atten_out), f22(atten_out), f23(atten_out), f24(atten_out), f25(atten_out), f26(atten_out) ])
            atten_in_effective=pif(freq)
            atten_out_effective=pof(freq)
        else:
            print("Wrong attenuator setting value range")
            return
    else:
        print("Wrong frequency range")
        return
    
    return atten_in_effective, atten_out_effective

if __name__ == "__main__":
    atten_in_effective, atten_out_effective=get_atten_SwitchBox(1e9, 52.25, 16.5)
    
