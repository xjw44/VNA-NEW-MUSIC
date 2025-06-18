# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:58:48 2023

@author: kids
"""


# import IQ_mixer_noise_funcs_rev5_noIQMixerCalib as IQfunc
import IQ_mixer_noise_funcs as IQfunc
from Hardware_Control import SIM960_PID
import time, os, h5py
from Hardware_Control import SIM921_RT as GRT
import numpy as np
import matplotlib.pyplot as plt


dir_path = 'D:/noise_data/IQ_data/'
# MB_filename = 'D:/YellowCryostatData/Be231102d2_AR_BS/Dark/20240214_11h46/Data/data_h5/Be231102d2_AR_BS_dark_processed.h5' 
V_KIDs = 3224 # Inductor volume, in um^3

# IQ_calibfileDC_path = 'D:/noise_data/IQ_calib/20220817_162615_DC/h5_data/IQ_calib_fit.h5'
# IQ_calibfileAC_path = 'D:/noise_data/IQ_calib/20220817_160302_AC/h5_data/IQ_calib_fit.h5'

IQ_calibfileDC_path = 'D:/noise_data/IQ_calib/20240325_133843_DC_gain20_RFm16/h5_data/IQ_calib_fit.h5'
IQ_calibfileAC_path = 'D:/noise_data/IQ_calib/20240325_144350_AC_gain200_RFm16/h5_data/IQ_calib_fit.h5'

# Main params
freq_sweep = True
# select IQ mixer: either 'SigaTek QD54A10' for a measurement between 200 MHz and 400 MHz, 
# or 'TSC AD0460B' for a measurement between 400 MHz and 2 GHz
IQ_mixer = 'SigaTek QD54A10' # either SigaTek QD54A10 or TSC AD0460B
source = ['dark'] #dark, Mirror, Hot, Cold
# dev_name = 'AlMn_Cf240501BR_ARandBS' #channel4
# cryo_chan = 4
# dev_name = 'NIST_device'
# cryo_chan = 2
dev_name = 'Be231102p2'
cryo_chan = 3
# dev_name = 'Be210504p2b1'
# cryo_chan = 1

FP_temp = None #None or [250, 280, 310, 340, 370, 400, 430] [250, 270, 290, 310, 330, 350]

# freq_center = np.array([1600e6]) #1208.182e6, 1273.208e6, 1208.074e6, 1272.853e6
# freq_center = np.array([260.12e6, 263.87e6, 276.28e6, 278.97e6]) # within 500 kHz of the real fres #sample:278.96e6, 326.31e6, 397.21e6 #255.83e6, 272.67e6, 323.72e6, 387.22e6
# freq_center = np.array([255.83e6, 272.67e6, 323.72e6, 387.22e6]) # within 500 kHz of the real fres #sample:278.96e6, 326.31e6, 397.21e6 #255.83e6, 272.67e6, 323.72e6, 387.22e6
# freq_center = np.array([213.08e6, 214.80e6, 247.41e6, 255.83e6, 256.86e6, 260.12e6, 263.86e6, 265.55e6, 269.67e6, 270.30e6, 272.67e6, 274.71e6, 276.38e6, 278.96e6, 308.52e6, 312.88e6]) # within 500 kHz of the real fres #sample:278.96e6, 326.31e6, 397.21e6 #255.83e6, 272.67e6, 323.72e6, 387.22e6
# freq_center = np.array([213.08e6, 214.80e6, 247.41e6, 255.83e6, 256.86e6, 260.12e6, 263.86e6, 265.55e6, 269.67e6, 
#                         270.30e6, 272.67e6, 274.71e6, 276.38e6, 278.96e6, 308.52e6, 312.88e6, 313.15e6, 318.56e6, 
#                         322.41e6, 323.72e6, 326.36e6, 335.79e6, 340.65e6, 368.18e6, 373.47e6, 377.84e6, 384.08e6, 
#                         387.22e6, 393.64e6, 397.18e6]) # within 500 kHz of the real fres #sample:278.96e6, 326.31e6, 397.21e6 #255.83e6, 272.67e6, 323.72e6, 387.22e6

# freq_center = np.array([213.08e6, 214.80e6, 247.41e6, 255.83e6, 256.86e6, 260.12e6, 263.86e6, 265.55e6, 269.67e6]) # within 500 kHz of the real fres #sample:278.96e6, 326.31e6, 397.21e6 #255.83e6, 272.67e6, 323.72e6, 387.22e6
# freq_center = np.array([270.30e6, 272.67e6, 274.71e6, 276.38e6, 278.96e6, 308.52e6, 312.88e6, 313.15e6, 318.56e6, 
#                         322.41e6, 323.72e6, 326.36e6, 335.79e6, 340.65e6, 368.18e6, 373.47e6, 377.84e6, 384.08e6]) # within 500 kHz of the real fres #sample:278.96e6, 326.31e6, 397.21e6 #255.83e6, 272.67e6, 323.72e6, 387.22e6
# freq_center = np.array([386.53e6, 386.87e6, 387.22e6, 393.64e6, 397.18e6, 401.25e6, 404.14e6, 409.10e6, 417.63e6, 504.12e6, 560.88e6, 596.45e6, 611.96e6, 628.62e6])

# freq_center = np.array([256.86e6, 270.30e6, 278.96e6, 326.36e6])
freq_center = np.array([255.7e6])
# freq_center = np.array([255.83e6, 256.86e6, 270.30e6, 278.96e6, 326.36e6]) # within 500 kHz of the real fres #sample:278.96e6, 326.31e6, 397.21e6 #255.83e6, 272.67e6, 323.72e6, 387.22e6
#Be231102p2: 213.00e6, 214.80e6, 247.41e6, 255.83e6, 256.86e6, 260.12e6, 263.86e6, 265.55e6, 269.67e6, 270.30e6, 272.67e6, 274.71e6, 276.38e6, 278.96e6, 308.52e6, 312.88e6, 313.15e6, 314.00e6, 314.39e6, 318.56e6, 322.41e6, 323.72e6, 326.36e6, 328.32e6, 328.71e6, 335.79e6, 340.65e6, 368.18e6, 373.47e6, 375.70e6, 375.84e6, 377.84e6, 384.08e6, 386.53e6, 386.87e6, 387.22e6, 393.64e6, 397.18e6, 401.25e6, 404.14e6, 409.10e6, 417.63e6, 504.12e6, 560.88e6, 596.45e6, 611.96e6, 628.62e6 

# freq_center = np.array([315.08e6])
#Be210504p2b1: 172.902e6, 177.499e6, 208.673e6, 209.238e6, 211.933e6, 215.182e6, 244.780e6, 246.228e6, 247.845e6, 251.390e6, 254.826e6, 258.301e6, 263.124e6, 264.553e6, 266.425e6, 271.605e6, 312.07e6, 312.4e6, 313.780e6, 315.08e6, 315.444e6, 319e6, 319.4e6, 322.574e6, 324.892e6, 325.671e6, 326.768e6, 327.643e6, 331.211e6, 332.878e6, 368.617e6, 371.640e6, 374.184e6, 375.591e6, 377.396e6, 381.510e6, 382.089e6, 384.974e6, 386.939e6, 391.857e6, 392.874e6, 394.665e6, 398.146e6, 399.593e6, 404.976e6, 405.601e6, 494.132e6, 642.762e6
#Be210504p2b1: 258.301e6, 271.605e6, 368.617e6, 371.640e6, 377.396e6, 382.089e6, 386.939e6, 394.665e6, 398.146e6

# freq_center = np.array([191.336e6, 222.470e6, 245.057e6, 278.212e6, 304.230e6, 335.097e6, 356.463e6, 371.862e6, 554.252e6, 601.235e6])
# freq_center = np.array([338.163e6, 340.270e6, 340.635e6, 344.430e6, 348.393e6, 360.950e6, 367.656e6, 379.942e6, 507.765e6, 572.478e6, 586.234e6, 586.910e6])
# # Be231102p1: 191.336e6, 196.219e6, 222.470e6, 230.366e6, 234.681e6, 238.420e6, 239.533e6, 240.456e6, 244.135e6, 245.057e6, 245.985e6, 249.769e6, 250e6, 278.212e6, 283.835e6, 284.056e6, 292.727e6, 293.826e6, 296.894e6, 304.230e6, 335.097e6, 338.163e6, 340.270e6, 340.635e6, 344.430e6, 348.393e6, 356.463e6, 360.950e6, 367.656e6, 371.862e6, 379.942e6, 507.765e6, 554.252e6, 572.478e6, 586.234e6, 586.910e6, 601.235e6
# # Be231102p1: 191.336e6, 222.470e6, 245.057e6, 278.212e6, 304.230e6, 335.097e6, 356.463e6, 371.862e6, 554.252e6, 601.235e6

# freq_center = np.array([200.592e6, 240.369e6, 270.444e6, 311.232e6])
# cf221001: 200.592e6, 240.369e6, 270.444e6, 311.232e6, 645.039e6

req_pwr_at_device = -60*np.ones(len(freq_center)) # dBm from -80 to -100
# req_pwr_at_device = np.array([-70, -70, -52])
pwr_sweep = np.array([0]) # dB, either None or a numpy array with the pwr sweep increments (0, -2, -4, etc.) 0, -4, -8, -12, -16, -20
# Weinschel_setchan2 = -12 # dB

# Wide freq sweep (to locate the resonance)
sweep_span = 10e6 # Hz
sweep_rate = 0.5 # Hz

# Frequency modulation measurement (to fit the resonance)
FM_span = 1e6 # Hz
FM_rate = 10 # Hz

# Calib noise measurement (using FM with a square modulation signal)
calib_rate = 100 # Hz

# Noise measurement

df_onoff_res = 0.5e6 # Hz, spacing between on and off resonance tones 3e6 #0.8e6 for two-tone test
N_noise = 2 # Number of averaged noise measurements

#Estimate time, single sampling time is 1 second
if FP_temp==None:
    ExpectedTime = len(source)*(len(freq_center)*len(pwr_sweep)*(56+30*N_noise)+18)/60/60
else:
    ExpectedTime = (len(source)*(len(freq_center)*len(pwr_sweep)*(56+30*N_noise))+500)*len(FP_temp)/60/60

print(f'The noise measurement expected time is {int(ExpectedTime)}h{int(np.round(ExpectedTime%1*60))}min')
input('Check the time and press enter to continue')

#%%
# time_start_1 = time.time()

if __name__ == "__main__":

    main_dict = {'IQ_mixer':IQ_mixer, 'source':source, 'FP_temp':FP_temp, 
              'dev_name':dev_name, 'cryo_chan':cryo_chan, 'dir_path':dir_path, 'pwr_sweep':pwr_sweep, 
              'freq_center':freq_center, 'req_pwr@device':req_pwr_at_device, 
              'sweep_span':sweep_span, 'sweep_rate':sweep_rate, 'FM_span':FM_span, 'FM_rate':FM_rate, 
              'calib_rate':calib_rate, 'df_onoff_res':df_onoff_res}    
    
    meas_dict = {}
    main_dict = IQfunc.init_measurement(main_dict)
    main_dict = IQfunc.define_atten_and_pwr(main_dict)
    main_dict = IQfunc.make_dirs(main_dict)
    
    data_dict, meas_dict = IQfunc.freq_sweep_main(main_dict, meas_dict)
    filename_h5 = [main_dict['time_hhmmss'] + '_IQmixer_res_%s.h5'%str for str in main_dict['freq_center_str']]
    
    rst = 1    
    for i_temp, temp in enumerate(main_dict['FP_temp']):
        if main_dict['FPtemp_sweep_bool']:
            with SIM960_PID.GRT_PID(temp*1e-3, rst=rst) as grt:
                grt.set_T()
            rst = 0
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
                    
                    # Set-up frequency sweep for resonance fit
                    meas_dict['mode'] = 'FM'
                    meas_dict['span'] = FM_span
                    meas_dict['rate'] = FM_rate
                    meas_dict['pwr@device'] = main_dict['pwr@device'][i_res][i_pwr]
                    meas_dict['pwr@device_str'] = ('%.0f'%(meas_dict['pwr@device'])).replace('-','m')
                    meas_dict['freq_center'] = main_dict['freq_center'][i_res]
                    meas_dict['freq_center_str'] = main_dict['freq_center_str'][i_res]
                    IQfunc.update_meas_dict(main_dict, meas_dict, i_res=i_res, i_pwr=i_pwr)

                    IQfunc.set_atten(meas_dict['freq_center'], meas_dict['atten_in'], meas_dict['atten_out'])

                    main_dict, meas_dict, SR560_dict, data_dict = IQfunc.FM_main(main_dict, meas_dict)
                    
                    resfit_dict = IQfunc.fit_res(data_dict)
                    IQfunc.save_data_FM(main_dict, meas_dict, SR560_dict, data_dict, resfit_dict, h5_filepath)
                    IQfunc.plot_fit_reso(data_dict, resfit_dict, main_dict, meas_dict)
                    
                    # Delta, alpha, f0 = IQfunc.get_MB_params(resfit_dict['fres'], temp, MB_filename)
                    
                    # Set-up frequency sweep for noise calib
                    meas_dict['mode'] = 'calib'
                    # meas_dict['FM_rate'] = calib_rate
                    meas_dict['rate'] = calib_rate
                    main_dict, meas_dict, SR560_dict, data_dict = IQfunc.take_calib(main_dict, meas_dict, resfit_dict, i_res)
                    calib_dict = IQfunc.calc_calib(data_dict, meas_dict)
                    IQfunc.save_data_calib(main_dict, meas_dict, SR560_dict, calib_dict, h5_filepath)
                    IQfunc.plot_calib_rotation(calib_dict, main_dict, meas_dict)
                    
                    # time_start_3 = time.time()
                    # print("each readout power running time: "+str(time_start_3-time_start_2)+" s")
                    # print('Start noise measurement')
                    # meas_dict['freq_center'] =  meas_dict['freq_center'] - 0.1e6
                    # print('change freq to %f'%meas_dict['freq_center'])
                    # input('Check the two tone setting to continue')
                    # Set-up noise measurement 
                    meas_dict['mode'] = 'noise'
                    # fig1, ax1, lines = IQfunc.plot_Sdff_ini(meas_dict)
                    Sdff_dict = {}
                    for i in range(N_noise):
                        print('i = %d'%i)
                        SG382, task = IQfunc.init_noise_measurement(main_dict, meas_dict)
                        noise_dict = IQfunc.noise_main(main_dict, meas_dict, SG382, task)
                        IQfunc.save_data_noise(main_dict, meas_dict, SR560_dict, noise_dict, h5_filepath, i)
                        noise_dict = IQfunc.rotate_noise(noise_dict, calib_dict)
                        noise_dict = IQfunc.calc_dff(noise_dict, calib_dict, main_dict, meas_dict)
                        Sdff_dict = IQfunc.calc_Sdff(meas_dict, noise_dict, Sdff_dict, i)
                        
                        SG382.write('ENBR 0')
                        task.close()
                    
                    Sdff_dict = IQfunc.average_Sdff(Sdff_dict, N_noise)
                    # SNqp = IQfunc.calc_SNqp(resfit_dict['fres'], temp, Sdff_dict, delta_0=Delta, V=V_KIDs)
                    # Sdff_dict['SNqp'] = SNqp
                    IQfunc.save_data_Sdff(main_dict, meas_dict, SR560_dict, Sdff_dict, h5_filepath, N_noise)
                    IQfunc.plot_Sdff(Sdff_dict, main_dict, meas_dict, ylims=[1e-20, 1e-14])
 
                    meas_dict['FPtemp_stop'] = float(GRT.GetTemp())
                    
                    # time_start_4 = time.time()
                    # print("each noise measurement running time: "+str(time_start_4-time_start_3)+" s")
          
    time.sleep(1) 
    with SIM960_PID.GRT_PID(0.2) as grt_PID:
        print('Turning off focal plane PID...')
        grt_PID.PID_rst()
    time.sleep(1)

