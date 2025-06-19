# -*- coding: utf-8 -*-
"""
Created on Wed 6/18/2025

@author: Robin XJW
"""

from Hardware_Control import CMT_VNA_functions as vna


def main():
    print("test hulo")
    vna.restart_app()
    vna.close_VNA()
    vna.setup_VNA(1, 1)
    f_center = 80*1e6 # hz 
    f_span = 1*1e6 # hz 
    data = vna.Measure_S21(f_center, f_span, "b1_pcb", 0)
    print(data)
    
if __name__ == "__main__":
    main()
