# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:58:48 2023

@author: kids
"""

import sys
import os

# Add the absolute path of the folder to sys.path
sys.path.append(os.path.abspath("D:/Python_Scripts/Noise_Measurement"))

# import IQ_mixer_noise_funcs_rev5_noIQMixerCalib_v1 as IQfunc
# import IQ_mixer_noise_funcs_rev5 as IQfunc
import IQ_mixer_noise_funcs_twotone as IQfunc
from Hardware_Control import SIM960_PID
import time, h5py
from Hardware_Control import SIM921_RT as GRT
import numpy as np
import matplotlib.pyplot as plt
# from Hardware_Control import SIM960_PID
from Hardware_Control.SDG2042X import SDGController
import IQ_mixer_process_twotones as Process2tones
#%%
dir_path = 'D:/noise_data/IQ_data/'
# MB_filename = 'D:/YellowCryostatData/Be231102d2_AR_BS/Dark/20240214_11h46/Data/data_h5/Be231102d2_AR_BS_dark_processed.h5' 
V_KIDs = 3224 # Inductor volume, in um^3

#IQ_calibfileDC_path = 'D:/noise_data/IQ_calib/20220817_162615_DC/h5_data/IQ_calib_fit.h5'
#IQ_calibfileAC_path = 'D:/noise_data/IQ_calib/20220817_160302_AC/h5_data/IQ_calib_fit.h5'

IQ_calibfileDC_path = 'D:/noise_data/IQ_calib/20240325_133843_DC_20_RFm16/h5_data/IQ_calib_fit.h5'
IQ_calibfileAC_path = 'D:/noise_data/IQ_calib/20240325_144350_AC_gain200_RFm16/h5_data/IQ_calib_fit.h5'

# Main params
freq_sweep = True
# select IQ mixer: either 'SigaTek QD54A10' for a measurement between 200 MHz and 400 MHz, 
# or 'TSC AD0460B' for a measurement between 400 MHz and 2 GHz
IQ_mixer = 'SigaTek QD54A10' # either SigaTek QD54A10 or TSC AD0460B
source = ['dark'] #dark, Mirror, Hot, Cold
# dev_name = 'AlMn_Cf240502BL_AR' #channel1
# cryo_chan = 1
# dev_name = 'AlMn_Cf240501BR_ARandBS' #channel4
# cryo_chan = 4
dev_name = 'Be231102d2'
cryo_chan = 4
# dev_name = 'AlMn_Cf240502BL_AR+BS'
# cryo_chan = 3

FP_temp = None #None or [250, 280, 310, 340, 370, 400, 430] [250, 270, 290, 310, 330, 350]


#AlMn_Cf240501BR_ARandBS : 136.317e6, 138.5911e6, 153.4413e6, 157.4383e6, 159.1437e6, 160.1898e6, 161.0471e6, 161.7925e6, 162.4172e6, 168.5829e6, 168.8166e6, 175.5778e6, 207.3990e6, 208.5255e6, 209.7557e6, 210.1018e6, 210.9594e6, 215.2416e6, 216.8010e6, 227.1982e6, 243.2314e6, 245.1355e6, 247.1737e6, 248.48e6, 
#AlMn_Cf240501BR_ARandBS :
# freq_center = np.array([1.363180e+08, 1.385920e+08, 1.534390e+08, 1.569680e+08,
#         1.586660e+08, 1.597090e+08, 1.605740e+08, 1.617600e+08,
#         1.619445e+08, 1.680760e+08, 1.687950e+08, 1.750580e+08,
#         2.073680e+08, 2.079250e+08, 2.091250e+08, 2.094640e+08,
#         2.109260e+08, 2.146000e+08, 2.161490e+08, 2.271650e+08,
#         2.425060e+08, 2.444140e+08, 2.464320e+08, 2.477340e+08,
#         2.480300e+08, 2.484830e+08, 2.498080e+08, 2.512110e+08,
#         2.531440e+08, 2.538230e+08, 2.546780e+08, 2.596500e+08,
#         2.620610e+08, 2.644870e+08])
    
# freq_center = np.array([1.363180e+08, 1.385920e+08,
#         1.586660e+08, 1.597090e+08, 1.605740e+08, 1.619445e+08, 
#         1.687950e+08, 1.750580e+08, 2.079250e+08, 2.109260e+08, 
#         2.512110e+08, 2.531440e+08, 2.538230e+08,
#         2.620610e+08, 2.644870e+08])

# freq_center = np.array([
#         1.687950e+08, 1.750580e+08,
#                 2.073680e+08, 2.079250e+08, 2.091250e+08, 2.094640e+08,
#                 2.109260e+08, 2.146000e+08, 2.161490e+08, 2.271650e+08,
#                 2.425060e+08, 2.444140e+08, 2.464320e+08, 2.477340e+08,
#                 2.484830e+08, 2.498080e+08, 2.512110e+08,
#                 2.531440e+08, 2.538230e+08, 2.546780e+08, 2.596500e+08,
#                 2.620610e+08, 2.644870e+08
#         ])

# freq_center = np.array([1.385920e+08,
#         1.687950e+08, 2.079250e+08, 
#         2.512110e+08,
#         2.644870e+08])

# freq_center = np.array([1.586660e+08, 1.619445e+08, 
#         2.109260e+08, 
#         2.538230e+08,
#         2.620610e+08])

# freq_center = np.array([212.94e+06])
# freq_center = np.array([2.5566e+08])
# freq_center = np.array([2.512110e+08])
# freq_center = np.array([312.72e6])
# freq_center = np.array([313.80e6])
freq_center = np.array([255.8e6])
# second_res = np.array([312.97e6])#only needed for 2 res measurement
second_res = np.array([314.33e6])#only needed for 2 res measurement

# freq_center = np.array([194.69e6])
# 213.00e6, 214.80e6, 247.41e6, 255.83e6, 256.86e6, 260.12e6, 263.86e6, 265.55e6, 269.67e6, 270.30e6, 272.67e6, 274.71e6, 276.38e6, 278.96e6, 308.52e6, 312.88e6, 313.15e6, 314.00e6, 314.39e6, 318.56e6, 322.41e6, 323.72e6, 326.36e6, 328.32e6, 328.71e6, 335.79e6, 340.65e6, 368.18e6, 373.47e6, 375.70e6, 375.84e6, 377.84e6, 384.08e6, 386.53e6, 386.87e6, 387.22e6, 393.64e6, 397.18e6, 401.25e6, 404.14e6, 409.10e6, 417.63e6, 504.12e6, 560.88e6, 596.45e6, 611.96e6, 628.62e6

# AlMn_Cf240502BL_AR:
# freq_center = np.array([1.57850e+08, 1.60881e+08, 1.84742e+08, 1.87256e+08, 1.87746e+08,
#         1.88236e+08, 1.90040e+08, 1.91481e+08, 1.94439e+08, 1.94944e+08,
#         1.96340e+08, 1.97742e+08, 1.98802e+08, 2.30748e+08, 2.42099e+08,
#         2.43266e+08, 2.43597e+08, 2.48357e+08, 2.50317e+08, 2.51929e+08,
#         2.53874e+08, 2.54098e+08, 2.82257e+08, 2.84546e+08, 2.85220e+08,
#         2.86865e+08, 2.88887e+08, 2.89211e+08, 2.91425e+08, 2.93466e+08,
#         2.97806e+08, 2.98465e+08, 2.99154e+08, 3.00001e+08, 3.00241e+08,
#         3.01508e+08, 3.07544e+08, 3.08198e+08])

# freq_center = np.array([1.57850e+08, 1.60881e+08, 1.84742e+08,
#        1.88236e+08, 1.90040e+08, 1.91481e+08,
#        1.97742e+08, 2.42099e+08, 2.43266e+08, 2.82257e+08, 
#        2.88887e+08, 2.91425e+08, 
#        2.97806e+08, 2.99154e+08, 3.00241e+08,
#        3.08198e+08])


# req_pwr_at_device = -40*np.ones(len(freq_center)) # dBm from -80 to -100
# req_pwr_at_device = np.array([-38-43+10]) # dBm from -80 to -100 #we have +43 amp before the cryo
req_pwr_at_device = np.array([-70]) # dBm from -80 to -100
# req_pwr_at_device = np.array([-70, -70, -52])
pwr_sweep = np.array([0]) # dB, either None or a numpy array with the pwr sweep increments (0, -2, -4, etc.) 0, -4, -8, -12, -16, -20
# Weinschel_setchan2 = -12 # dB

# Wide freq sweep (to locate the resonance)
sweep_span = 5e6 # Hz

sweep_rate = 0.5 # Hz

# Frequency modulation measurement (to fit the resonance)
FM_span = 0.35e6 # Hz
FM_rate = 10 # Hz

# Calib noise measurement (using FM with a square modulation signal)
calib_rate = 10 # Hz

# Noise measurement

df_onoff_res = 0.2e6 # Hz, spacing between on and off resonance tones
N_noise = 10 # Number of averaged noise measurements

#Estimate time, single sampling time is 1 second with N_noise = 50
if FP_temp==None:
    ExpectedTime = len(source)*(len(freq_center)*len(pwr_sweep)*(56+30*N_noise)+18)/60/60
else:
    ExpectedTime = (len(source)*(len(freq_center)*len(pwr_sweep)*(56+30*N_noise))+500)*len(FP_temp)/60/60

SweepMode = 'AC'
# f1 = np.array([100e3]) #
# F_DAQ = 156250*2
# BW = 30e3/5 #kHz
Filter_order = 2


F_DAQ = 20e6/(4*5)
sdg = SDGController()
awg_power = 0 #dbm

def get_2res_sdg_vals(r1, r2):
    dif = abs(r1-r2)
    assert dif<F_DAQ/2, 'The difference between the two resonances is larger than the Nyquist frequency'
    mod_tone = dif/2
    main_tone = F_DAQ/4
    f0 = np.array([(main_tone-mod_tone),(main_tone+mod_tone)]) #kHz
    return f0

if SweepMode == '2_Res':
    f0 = get_2res_sdg_vals(freq_center[0],  second_res[0])
else:
    f0 = np.array([(250-120)*1e3,(250+120)*1e3]) #kHz
sdg.start_up_awg_dsb_modulation(f0, awg_power)
# f1 = np.array([70*1e3,4*70*1e3]) #kHz
# f1 = np.array([175e3])
# f1 = np.array([50e3])
df_onoff_res = f0[1]-f0[0]
f2 = reversed(f0)
F_DAQ = 20e6/(4*5)
BW = 30e3/30 #kHz

F_sampling = F_DAQ
BandWidth = BW
CarrierTones = f0


print(f'The noise measurement expected time is {int(ExpectedTime)}h{int(np.round(ExpectedTime%1*60))}min')
# input('Check the time and press enter to continue')

#%%
# time_start_1 = time.time()

if __name__ == "__main__":

    main_dict = {'IQ_mixer':IQ_mixer, 'source':source, 'FP_temp':FP_temp, 
              'dev_name':dev_name, 'cryo_chan':cryo_chan, 'dir_path':dir_path, 'pwr_sweep':pwr_sweep, 
              'freq_center':freq_center, 'req_pwr@device':req_pwr_at_device, 
              'sweep_span':sweep_span, 'sweep_rate':sweep_rate, 'FM_span':FM_span, 'FM_rate':FM_rate, 
              'calib_rate':calib_rate, 'df_onoff_res':df_onoff_res, 'F_sampling':F_sampling, 'SweepMode':SweepMode, 'second_res':second_res,
              'Modulation_Tones':f0}    
    
    meas_dict = {}
    main_dict = IQfunc.init_measurement(main_dict)
    main_dict = IQfunc.define_atten_and_pwr(main_dict)
    main_dict = IQfunc.make_dirs(main_dict)

    
    
    
    data_dict, meas_dict = IQfunc.freq_sweep_main(main_dict, meas_dict, f0, F_DAQ, BW, Filter_order, SweepMode) #make the sweep
    
   


    filename_h5 = [main_dict['time_hhmmss'] + '_IQmixer_res_%s.h5'%str for str in main_dict['freq_center_str']]
    
    rst = 1    
    for i_temp, temp in enumerate(main_dict['FP_temp']):
        # if main_dict['FPtemp_sweep_bool']:
        #     with SIM960_PID.GRT_PID(temp*1e-3, rst=rst) as grt:
        #         grt.set_T()
        #     rst = 0
        # time_start_2 = time.time()
        # print("PID running time: "+str(time_start_2-time_start_1)+" s")
        for i_BBT in range(len(source)):
            # meas_dict['source'] = source[int((i_temp+i_BBT)%2)]
            meas_dict['source'] = source[i_BBT]
            print('Start source=%s measurement'%meas_dict['source'])
            # input('Check the source and press enter to continue')
            for i_res, res in enumerate(freq_center):
                h5_filepath = main_dict['h5_dir'] + filename_h5[i_res]
                
                for i_pwr, pwr in enumerate(main_dict['pwr@device'][i_res]):
                    meas_dict['FPtemp_start'] = float(GRT.GetTemp())  
                    
                    # input('Check DC setting to continue')
                    # Set-up frequency sweep for resonance fit
                    if SweepMode == '2_Res':
                        f0 = get_2res_sdg_vals(main_dict['freq_center'][i_res], main_dict['second_res'][i_res])
                        sdg.start_up_awg_dsb_modulation(f0, awg_power)
                        f2 = reversed(f0)
                        second_main_dict = main_dict.copy()
                    meas_dict['mode'] = 'FM'
                    meas_dict['span'] = FM_span
                    meas_dict['rate'] = FM_rate
                    meas_dict['pwr@device'] = main_dict['pwr@device'][i_res][i_pwr]
                    meas_dict['pwr@device_str'] = ('%.0f'%(meas_dict['pwr@device'])).replace('-','m')
                    meas_dict['freq_center'] = main_dict['freq_center'][i_res]
                    meas_dict['freq_center_str'] = main_dict['freq_center_str'][i_res]
                    if SweepMode == '2_Res':
                        second_meas_dict = meas_dict.copy()
                        second_meas_dict['freq_center'] = second_main_dict['freq_center'][i_res]
                        second_meas_dict['freq_center_str'] = main_dict['freq_center_str'][i_res] 
                    IQfunc.update_meas_dict(main_dict, meas_dict, i_res=i_res, i_pwr=i_pwr)

                    IQfunc.set_atten(meas_dict['freq_center'], meas_dict['atten_in'], meas_dict['atten_out'])
                    
                    main_dict, meas_dict, SR560_dict, data_dict = IQfunc.FM_main(main_dict, meas_dict, f0, F_DAQ, BW, Filter_order, SweepMode = 'AC')
                    
                    resfit_dict = IQfunc.fit_res(data_dict, f0)
                    IQfunc.save_data_FM(main_dict, meas_dict, SR560_dict, data_dict, resfit_dict, h5_filepath)
                    fit_res_fig, fit_res_axs = IQfunc.plot_fit_reso(resfit_dict, main_dict, meas_dict)
                    if SweepMode == '2_Res': #if you're fiting for the second tones, then you need to take data from the second tone for demodulation
                        second_main_dict, second_meas_dict, SR560_dict, second_data_dict = IQfunc.FM_main(main_dict.copy(), meas_dict.copy(), f0, F_DAQ, BW, Filter_order, SweepMode = '2_Res')
                        offset_h5_filepath = main_dict['h5_dir'] + "off_res" +  filename_h5[i_res]
                        second_resfit_dict = IQfunc.fit_res(second_data_dict, f2) 
                        IQfunc.save_data_FM(second_main_dict, second_meas_dict, SR560_dict, second_data_dict, second_resfit_dict, offset_h5_filepath)
                        IQfunc.plot_fit_reso(second_resfit_dict, second_main_dict, second_meas_dict)
                        
                        Fres1 = resfit_dict['fres']
                        Fres2 = second_resfit_dict['fres']
                        f0 = get_2res_sdg_vals(resfit_dict['fres'], second_resfit_dict['fres'])
                        f2 = reversed(f0)
                        sdg.start_up_awg_dsb_modulation(f0, awg_power)

                        
                        # Then double check that you get the same values from fitting f1
                        main_dict_check, meas_dict_check, SR560_dict, data_dict_check = IQfunc.FM_main(main_dict.copy(), meas_dict.copy(), f0, F_DAQ, BW, Filter_order, SweepMode = 'AC')
                        
                        resfit_dict_check = IQfunc.fit_res(data_dict_check, f0)
                        assert abs(Fres1-resfit_dict_check['fres'])<1e4, "Resonance is not consistent after changing mod frequency"
                        
                        second_main_dict_check, second_meas_dict_check, SR560_dict, second_data_dict_check = IQfunc.FM_main(second_main_dict.copy(), second_meas_dict.copy(), f0, F_DAQ, BW, Filter_order, SweepMode = '2_Res')
                        
                        second_resfit_dict_check = IQfunc.fit_res(second_data_dict_check, f2)
                        
                        assert abs(Fres2-second_resfit_dict_check['fres'])<1e4
                        

                        

                    
                    # Set-up frequency sweep for noise calib
                    meas_dict['mode'] = 'calib'
                    # meas_dict['FM_rate'] = calib_rate
                    meas_dict['rate'] = calib_rate
                    main_dict['Modulation_Tones'] = f0

                    main_dict, meas_dict, SR560_dict, calib_data_dict = IQfunc.take_calib(
                        main_dict, meas_dict, resfit_dict, i_res, F_DAQ, f0, SweepMode = 'AC')
                    calib_dict = IQfunc.calc_calib(calib_data_dict, meas_dict, main_dict, f0, BW, Filter_order = Filter_order, SweepMode = 'AC')
                    IQfunc.save_data_calib(main_dict, meas_dict, SR560_dict, calib_dict, h5_filepath)
                    IQfunc.plot_calib_rotation(calib_dict, main_dict, meas_dict, resfit_dict, fit_res_fig, fit_res_axs)
                    calib_dict['angle'] = resfit_dict['angle']
                    if SweepMode == '2_Res':
                        second_meas_dict['mode'] = 'calib'
                        second_meas_dict['rate'] = calib_rate
                        second_main_dict, second_meas_dict, SR560_dict, second_data_dict = IQfunc.take_calib(
                            second_main_dict, second_meas_dict, second_resfit_dict, i_res, F_DAQ, f2, SweepMode = 'AC')
                        second_calib_dict = IQfunc.calc_calib(second_data_dict, second_meas_dict, second_main_dict, f2, BW, Filter_order, SweepMode = 'AC')
                        IQfunc.save_data_calib(second_main_dict, second_meas_dict, SR560_dict, second_calib_dict, offset_h5_filepath)
                        IQfunc.plot_calib_rotation(second_calib_dict, second_main_dict, second_meas_dict, second_resfit_dict)
                    # time_start_3 = time.time()
                    # print("each readout power running time: "+str(time_start_3-time_start_2)+" s")
                    
                    print('Start two tone noise measurement')
                    meas_dict['freq_center'] =  meas_dict['freq_center'] #- f1[0]

                    meas_dict['mode'] = 'noise'
                    Sdff_dict = {}
                    
                    for i in range(N_noise):
                        print('i = %d'%i)
                        SG382, task = IQfunc.init_noise_measurement(main_dict, meas_dict)
                        noise_dict = IQfunc.noise_main(main_dict, meas_dict, SG382, task)
                        IQfunc.save_data_noise(main_dict, meas_dict, SR560_dict, noise_dict, h5_filepath, i)
                        SG382.write('ENBR 0')
                        task.close()

                    if SweepMode == 'AC':
                        Idemo,Qdemo = Process2tones.process_twotonedata_twotonemode(main_dict['dir_path'], h5_filepath, I_demodulated = False, Q_demodulated = False)
                    elif SweepMode == '2_Res':
                        Process2tones.process_twotonedata_TwoResMode(main_dict['dir_path'], h5_filepath)  
                        
                    IQfunc.plot_noise(calib_dict, noise_dict, timestream_factor = 0.01, decimation_factor = 1,I_demodulated = Idemo, Q_demodulated = Qdemo)
          
    time.sleep(1) 
    with SIM960_PID.GRT_PID(0.2) as grt_PID:
        print('Turning off focal plane PID...')
        grt_PID.PID_rst()
    time.sleep(1)

