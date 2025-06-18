# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:23:50 2019

@author: F Defrance

This class commands SIM960_PID that controls the focal plane temperature of the yellow cryostat
A specific care should be taken to handle errors correctly since the PID, resistance bridge (SIM921), and multiplexer units all she the same frame
(SIM900) and the same GPIB address. Therefore, when established, the connection is often lost (maybe because Labview communicates with the 
another instrument of SIM900). Therefore, before each set of commands it is necessary to check the connection and the identity of the instrument
we are talking to. 

Example of commands using the PID:
1) To reset the PID
with GRT_PID() as grt_PID:
    grt_PID.PID_rst()

2) To fix a setpoint to the PID (the PID will then try to reach this temperature, but it won' t be monitored by the program)
(0.285 K is the setpoint, rst indicates is we want to reset the PID before starting it)
with GRT_PID(0.285, rst = 1) as grt_PID:
    grt_PID.PID_set()
    
3) To know the current PID output
with GRT_PID() as grt_PID:
    print(grt_PID.PID_Vout())
    
4) To fix a setpoint to the PID and wait until the temperature has been reached
(0.285 K is the setpoint, rst indicates is we want to reset the PID before starting it)
with GRT_PID(0.285, rst = 1) as grt_PID:
    grt_PID.set_T()

A class has been used to automatically disconnect and close the connection at the end of each request 

"""

import time
import pyvisa
from Hardware_Control import SIM921_RT


# def set_PID_params():
#     P = -1
#     I = 0.4 # 0.25 (old values)
#     D = 0.1
#     Vout_min = 0.07 # in [V]. To correct the offset of the PID which is approx -0.1
#     Vout_max = 5 # in [V]. The max limit used to be 2.7V, but we want to be able to reach higher temp
#     rate = 0.01
#     params = {} # Create dict with PID parameters
#     for i in ('P', 'I', 'D', 'Vout_min', 'Vout_max', 'rate'):
#         params[i] = locals()[i]
#     return params

    
# def PID_ini():
#     # Initialize the PID (maybe all the lines are not necessary)
#     delay_time = 0.1
#     rm = pyvisa.ResourceManager()
#     inst = rm.open_resource('GPIB0::5::INSTR')
#     inst.read_termination = '\n'
#     inst.write_termination = '\n'
#     inst.write('xyzzy')
#     time.sleep(delay_time)
#     inst.write('CONN 3,"xyzzy"')
#     time.sleep(delay_time)
#     inst.write("CHAN 4")
#     #print("\nSetting Channel to 4\n")
#     time.sleep(delay_time)
#     inst.write('xyzzy')
#     time.sleep(delay_time)
#     inst.write('CONN 4,"xyzzy"')
#     time.sleep(delay_time)
#     return inst
    
    
# def PID_rst(inst):
#     """
#     Reset the PID
#     So we flush the PID memory 
#     and we make sure the PID always runs with the same params
#     """
#     delay_time = 0.1
#     inst.write('*RST')
#     time.sleep(delay_time)
#     inst.write('AMAN 0')
#     time.sleep(delay_time)
#     inst.write('MOUT 0')
#     time.sleep(delay_time)
  
    
# def PID_conf(rst = 0):
#     """
#     Set the PID params (needed especially after reset)
#     The rst option allows to trigger or not the reset command
#     """
#     inst = PID_ini()
#     if rst: PID_rst(inst)
#     delay_time = 0.1
#     params = set_PID_params()
#     inst.write('GAIN {}'.format(params['P']))
#     time.sleep(delay_time)
#     inst.write('INTG {}'.format(params['I']))
#     time.sleep(delay_time)
#     inst.write('DERV {}'.format(params['D']))
#     time.sleep(delay_time)
#     inst.write('PCTL 1')
#     time.sleep(delay_time)
#     inst.write('ICTL 1')
#     time.sleep(delay_time)
#     inst.write('DCTL 1')
#     time.sleep(delay_time)
#     inst.write('RATE {}'.format(params['rate']))
#     time.sleep(delay_time)
#     inst.write('LLIM {}'.format(params['Vout_min']))
#     time.sleep(delay_time)
#     inst.write('ULIM {}'.format(params['Vout_max']))
#     time.sleep(delay_time)
#     inst.write('INPT 0')
#     time.sleep(delay_time)
#     return inst


# def PID_set(aimT, rst=0):
#     """
#     Set the target temperature (aimT, in [K]) of the PID
#     The try/except structure prevent I/O collision in the instrument to stop the program
#     The rst option allows to trigger or not the reset command
#     """
#     visaerror = 1
#     while visaerror == 1:
#         try:
#             inst = PID_conf(rst = rst)
#             aimR = round(SIM921_RT.T2R_interp(aimT))/1000
#             print('Desired temp: %.1f mK (R = %d Ohms)'%(aimT*1e3, aimR*1e3))
#             inst.write('SETP {:1.3f}'.format(aimR))  
#             inst.write('SETP?')
#             Rset = float(inst.read())
#             while Rset != aimR:
#                 inst = PID_ini()
#                 inst.write('SETP {:1.3f}'.format(aimR))   
#                 inst.write('SETP?')
#                 Rset = inst.read()
#                 print(Rset, aimR)
                
#             if Rset == aimR:
#                 # If the resistance value set in the PID is the same as aimR
#                 # we can switch the output from manual to PID
#                 inst.write('AMAN 1') 
#                 print('PID started!')
#             else: print('Error: Rset different from aimR!')
#             visaerror = 0
#         except pyvisa.VisaIOError:
#             print("VisaIOError... Let's try again!")
#             visaerror = 1
#     return inst
        
        
# def PID_Vout():
#     """
#     Read the voltage at the output of the PID (Vout in [V])
#     """
#     visaerror = 1
#     while visaerror == 1:
#         try:
#             inst = PID_ini()  
#             inst.write("OMON?")
#             Vout =  float(inst.read()) 
#             visaerror = 0
#         except pyvisa.VisaIOError:
#             print("VisaIOError... Let's try again!")
#             visaerror = 1
#     return Vout
        

# def set_T(aimT, rst = 0):
#     """
#     Defines the tolerance (interval within whitch the temperature is considered as close enough to aimT)
#     Defines max_dev, the maximum temperature variation between 2 consecutive measurements
#     If the measurement fulfills the tolerance and max_dev requirements 4 times in a row
#     (with 1 measurement every 30s), the aimT temperature (in [K]) is considered reached and the PID stops
#     """
#     PID_set(aimT, rst = rst)
#     tolerance = max(abs((240-aimT*1e3)/100)**3*1e-3, 0.8e-3)
#     max_dev = max((aimT/600)**2*1e3, 0.3e-3)
#     print('Tolerance = %.1f mK\nMax temperature variation = %.1f mK\n'%(tolerance*1e3, max_dev*1e3))
#     nGoodLoops = 0
#     temp0 = 0
#     time.sleep(1)
#     while nGoodLoops < 4:
#         temp1 = temp0
#         temp0 = SIM921_RT.GetTemp()  
#         Vout = PID_Vout()
#         print('Goal: %.1f mK. Current: %.1f mK. PID Vout = %.3f V'%(aimT*1e3, temp0*1e3, Vout))

#         if (abs(temp1-temp0) < (max_dev)) and (abs(temp0-aimT)<tolerance):
#             nGoodLoops = nGoodLoops+1
#         else:
#             nGoodLoops = 0
            
#         time.sleep(30)

        
class GRT_PID:
    def __init__(self, aimT=0, rst=0):
        self.aimT = aimT
        self.rst = rst
        self.delay_time = 0.05
        self.long_delay = 1
        # self.rm = None
        # self.inst = None
        
    def __enter__(self):
        """ Connects to the PID (SIM960)
        """
        self.PID_connect()
        return self
            

    def PID_connect(self):
        """
        Connects to the PID. 
        It uses GPIB address 5, 
        the different commands are used to establish the connection with SIM960
        At the end, we query the identity of the instrument and make sure we talk with the correct one.
        Each SIM960 potentially has a different name (maybe different firmware)
        Previous SIM960 was Stanford_Research_Systems,SIM960,s/n006629,ver2.17
        current one is Stanford_Research_Systems,SIM960,s/n003169,ver2.12
        If the identity is wrong, we do all the connection procedure again, until it gets right
        """
        print('PID_connect')
        while True:
            try: 
                while True:
                    self.rm = pyvisa.ResourceManager()
                    self.inst = self.rm.open_resource('GPIB0::5::INSTR')
                    self.inst.read_termination = '\n'
                    self.inst.write_termination = '\n'
                    self.inst.write('xyzzy')
                    time.sleep(self.delay_time)
                    self.inst.write('CONN 3,"xyzzy"')
                    time.sleep(self.delay_time)
                    self.inst.write("CHAN 4")
                    time.sleep(self.delay_time)
                    self.inst.write('xyzzy')
                    time.sleep(self.delay_time)
                    self.inst.write('CONN 4,"xyzzy"')
                    time.sleep(self.delay_time)
                    self.inst.write('*IDN?')
                    idn = self.inst.read()     
#                    if idn == 'Stanford_Research_Systems,SIM960,s/n006629,ver2.17\r':
                    if idn == 'Stanford_Research_Systems,SIM960,s/n003169,ver2.12\r':
                        break
                    else:
                        print('Error! Expected "Stanford_Research_Systems,SIM960,s/n003169,ver2.12", got ')
#                        print('Error! Expected "Stanford_Research_Systems,SIM960,s/n006629,ver2.17", got ')
                        print(idn)
                        print("Let's connect again")
                        time.sleep(self.long_delay)
                break
            except pyvisa.VisaIOError:
                print("VisaIOError1... Let's try again!")
                time.sleep(self.long_delay)
                    
    def PID_rst(self):
        """
        Reset the PID
        So we flush the PID memory 
        and we make sure the PID always runs with the same params
        """
        self.inst.write('*RST')
        time.sleep(self.delay_time)
        self.inst.write('AMAN 0')
        time.sleep(self.delay_time)
        self.inst.write('MOUT 0')
        time.sleep(self.delay_time)
      
    def PID_conf(self):
        """
        Set the PID params (needed especially after reset)
        The rst option allows to trigger or not the reset command
        """
        if self.rst: self.PID_rst()
        self.set_PID_params()
        self.inst.write('GAIN {}'.format(self.params['P']))
        time.sleep(self.delay_time)
        self.inst.write('INTG {}'.format(self.params['I']))
        time.sleep(self.delay_time)
        self.inst.write('DERV {}'.format(self.params['D']))
        time.sleep(self.delay_time)
        self.inst.write('PCTL 1')
        time.sleep(self.delay_time)
        self.inst.write('ICTL 1')
        time.sleep(self.delay_time)
        self.inst.write('DCTL 1')
        time.sleep(self.delay_time)
        self.inst.write('RATE {}'.format(self.params['rate']))
        time.sleep(self.delay_time)
        self.inst.write('LLIM {}'.format(self.params['Vout_min']))
        time.sleep(self.delay_time)
        self.inst.write('ULIM {}'.format(self.params['Vout_max']))
        time.sleep(self.delay_time)
        self.inst.write('INPT 0')
        time.sleep(self.delay_time)
            
    def set_PID_params(self):
        """
        Defines the PID parameters

        """
        P = -4#-1 
        I = 0.1#0.2#0.4 # 0.25 (old values)
        D = 0#0.2 #0.1
        Vout_min = 0#0.07 # in [V]. To correct the offset of the PID which is approx -0.1
        Vout_max = 5 # in [V]. The max limit used to be 2.7V, but we want to be able to reach higher temp
        rate = 0.01
        self.params = {} # Create dict with PID parameters
        for i in ('P', 'I', 'D', 'Vout_min', 'Vout_max', 'rate'):
            self.params[i] = locals()[i]
            
    def PID_set(self):
        """
        Set the target temperature (aimT, in [K]) of the PID
        The try/except structure prevent I/O collision in the instrument to stop the program
        The rst option allows to trigger or not the reset command
        """
        while True:
            try:
                self.PID_conf()
                aimR = round(SIM921_RT.T2R_interp(self.aimT))/1000
                print('Desired temp: %.1f mK (R = %d Ohms)'%(self.aimT*1e3, aimR*1e3))
                self.inst.write('SETP {:1.4f}'.format(aimR))  
                self.inst.write('SETP?')
                Rset = float(self.inst.read())
                while Rset != aimR:
                    self.PID_connect()
                    self.inst.write('SETP {:1.4f}'.format(aimR))   
                    self.inst.write('SETP?')
                    Rset = self.inst.read()
                    # print(Rset, aimR)
                if Rset == aimR:
                    time.sleep(self.long_delay)
                    # print(self.PID_Vout())
                    # If the resistance value set in the PID is the same as aimR
                    # we can switch the output from manual to PID
                    self.inst.write('AMAN 1') 
                    time.sleep(self.long_delay)
                    # print(self.PID_Vout())
                    print('PID started!')
                else: print('Error: Rset different from aimR!')
                break
            except pyvisa.VisaIOError:
                print("VisaIOError2... Let's try again!")
                print("PID_connect again")
                self.PID_connect()
                time.sleep(self.long_delay)
                
    def PID_Vout(self):
        """
        Read the voltage at the output of the PID (Vout in [V])
        """
        while True:
            try:
                self.inst.write("OMON?")
                Vout = float(self.inst.read()) 
                break
            except pyvisa.VisaIOError:
                print("VisaIOError3... Let's try again!")
                print("PID_connect again")
                self.PID_connect()
                time.sleep(self.long_delay)
        return Vout
            
    def set_T(self):
        """
        Defines the tolerance (interval within whitch the temperature is considered as close enough to aimT)
        Defines max_dev, the maximum temperature variation between 2 consecutive measurements
        If the measurement fulfills the tolerance and max_dev requirements 4 times in a row
        (with 1 measurement every 30s), the aimT temperature (in [K]) is considered reached and the PID stops
        """
        self.PID_set()
        tolerance = max(abs((200-self.aimT*1e3)/100)**2*1e-3, 1e-3)
        max_dev = max((self.aimT/800)**2*1e3, 0.15e-3)
        print('Tolerance = %.1f mK\nMax temperature variation = %.1f mK\n'%(tolerance*1e3, max_dev*1e3))
        nGoodLoops = 0
        temp0 = 0
        time.sleep(1)
        while nGoodLoops < 4:
            temp1 = temp0
            self.PID_disconnect()
            temp0 = SIM921_RT.GetTemp() 
            self.PID_connect()
            # print('connected')
            Vout = self.PID_Vout()
            print('Goal: %.1f mK. Current: %.1f mK. PID Vout = %.3f V'%(self.aimT*1e3, temp0*1e3, Vout))
            self.inst.write('SETP?')
            Rset = float(self.inst.read())
            Rcur = SIM921_RT.getRes(1)
            print('Goal: %.3f, actual: %.3f'%(Rset, Rcur))
            if (abs(temp1-temp0) < (max_dev)) and (abs(temp0-self.aimT)<tolerance):
                nGoodLoops = nGoodLoops+1
            else:
                nGoodLoops = 0
            if Vout < 0:
                time.sleep(10)
                self.PID_connect()
#                self.inst.write('MOUT {}'.format(self.params['Vout_min']))
#                print(SIM921_RT.GetTemp() )
                self.PID_rst()
                self.PID_set()
                nGoodLoops = 0
            time.sleep(30)


    def monitor_PID(self):
        """
        (with 1 measurement every 30s), the aimT temperature (in [K]) is considered reached and the PID stops
        """
        # self.PID_set()
        # tolerance = max(abs((200-self.aimT*1e3)/100)**2*1e-3, 1e-3)
        # max_dev = max((self.aimT/800)**2*1e3, 0.15e-3)
        # print('Tolerance = %.1f mK\nMax temperature variation = %.1f mK\n'%(tolerance*1e3, max_dev*1e3))
        nGoodLoops = 0
        temp0 = 0
        time.sleep(1)
        while nGoodLoops < 4:
            temp1 = temp0
            self.PID_disconnect()
            temp0 = SIM921_RT.GetTemp() 
            self.PID_connect()
            # print('connected')
            Vout = self.PID_Vout()
            print('Goal: %.1f mK. Current: %.1f mK. PID Vout = %.3f V'%(self.aimT*1e3, temp0*1e3, Vout))
            self.inst.write('SETP?')
            Rset = float(self.inst.read())
            Rcur = SIM921_RT.getRes(1)
            print('Goal: %.3f, actual: %.3f'%(Rset, Rcur))
#             if (abs(temp1-temp0) < (max_dev)) and (abs(temp0-self.aimT)<tolerance):
#                 nGoodLoops = nGoodLoops+1
#             else:
#                 nGoodLoops = 0
#             if Vout < 0:
#                 time.sleep(10)
#                 self.PID_connect()
# #                self.inst.write('MOUT {}'.format(self.params['Vout_min']))
# #                print(SIM921_RT.GetTemp() )
#                 self.PID_rst()
#                 self.PID_set()
#                 nGoodLoops = 0
            time.sleep(30)
    
    
    
    def PID_disconnect(self):
        print('PID_disconnect')
        self.inst.close()
        self.rm.close()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('__exit__ called')
        self.PID_disconnect()
#        self.inst.close()
#        self.rm.close()
        
    
# List of the values changed by reset command, and their values in () as they were initially set
#DISX ON
#DISP PRP (IGL now)
#SHFT OFF
#GAIN 1.0
#APOL POS  (NEG 0 currently)
#INTG 1.0
#DERV 1.0E-6
#OFST 0.0
#RATE 1.0 (+1.0E-2)
#PCTL ON
#ICTL OFF (1)
#DCTL OFF (1)
#OCTL OFF (0)
#RAMP OFF
#SETP 0 (+1.874)  (must not precedeRAMPOFF)
#MOUT 0.0 (-0.1)
#ULIM +10.0 (2.7)
#LLIM -10.0 (0.1)
#INPT EXT (0)
#AMAN PID 
#TOKN OFF
#SOUT

if __name__ == "__main__":
# To set a temperature. rst should be = to 1 for the first value (to reset the PID), then to 0 for the next values of the sweep
    with GRT_PID(0.260, rst=1) as grt:
        # grt.set_T()
        grt.monitor_PID()

# To stop and reset the PID
# with GRT_PID(0.275, rst = 1) as grt_PID:
#     grt_PID.PID_rst()

# Get Vout
# with GRT_PID(0.275, rst = 0) as grt_PID:
#     print(grt_PID.PID_Vout())
#    
#    
#    print(grt_PID.PID_Vout())
#    grt_PID.set_T()
#    grt_PID.inst.write('AMAN 1')
#    print(float(grt_PID.inst.read()))
#    grt_PID.PID_set()
#    grt_PID.PID_conf()
#    time.sleep(0.5)
#    print(grt_PID.PID_Vout())
#    grt_PID.inst.write('SETP 1.151')  
#    time.sleep(0.5)
#    print(grt_PID.PID_Vout())
#    
#    grt_PID.inst.write('AMAN 1')
#    time.sleep(1)
#    print(grt_PID.PID_Vout())
#    time.sleep(1)
#    print(grt_PID.PID_Vout())
#    time.sleep(5)
#    print(grt_PID.PID_Vout())
#    time.sleep(5)
#    print(grt_PID.PID_Vout())
#    time.sleep(5)
#    print(grt_PID.PID_Vout())    
#    time.sleep(5)
    #    grt_PID.inst.write('LLIM 0')
#    print(grt_PID.inst.read())
#    grt_PID.inst.write('LLIM ?')
#    print(grt_PID.inst.read())
#    grt_PID.PID_rst()
#    grt_PID.PID_conf()
#    print(grt_PID.PID_Vout())
#    grt_PID.set_T()        
#    grt_PID.inst.write('SETP?')
#    Rset = float(grt_PID.inst.read())
#    Rcur = SIM921_RT.getRes(1)
#    print('Goal: %.3f, actual: %.3f'%(Rset, Rcur))
    
    

    
    
