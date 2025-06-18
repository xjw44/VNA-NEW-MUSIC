# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:47:42 2021

@author: Fabien D
To set the BB PID independently
"""

import queue, time
import set_BB_temp as setBBT


def set_BB_PID(BB_PID):
    if BB_PID["mode"] == 'start':
        setBBT.Tref = BB_PID["Tref"]
        setBBT.stable_flag = False 
        BB_PID["que"] = queue.Queue()
        BB_PID["job"] = setBBT.PID_BBT(BB_PID["que"]) 
        BB_PID["job"].start()
        print("--- Tref = %.3f K, PID starting ---"%BB_PID["Tref"])
    elif BB_PID["mode"] == 'update':
        setBBT.Tref = BB_PID["Tref"]
        setBBT.stable_flag = False 
        print("--- New Tref = %.3f K ---"%BB_PID["Tref"])
    elif BB_PID["mode"] == 'wait':
        while setBBT.stable_flag == False:
            time.sleep(10)
        if setBBT.stable_flag == True:
            print("--- Tref reached! ---")
        else: 
            print("Error 007! something is wrong!")  
    elif BB_PID["mode"] == 'stop':
        BB_PID["job"].stop()
        time.sleep(5)
        #BB_PID["que"].join()
        # while not BB_PID["que"].empty():
        #     BB_PID["BBT_PID_data"] = BB_PID["que"].get() 
    return BB_PID


# Commands to use to set the PID
# To start the PID and set a temperature
BB_PID = {"mode":'start', "Tref":8}
BB_PID = set_BB_PID(BB_PID)  

# To update the PID and set a new temperature 
BB_PID = {"mode":'update', "Tref":6}
BB_PID = set_BB_PID(BB_PID)  

# To Stop the PID at the end of the measurement
BB_PID["mode"] = 'stop'
BB_PID = set_BB_PID(BB_PID)     