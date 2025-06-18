# -*- coding: utf-8 -*-
"""
Created on Wed 6/18/2025

@author: Robin XJW
"""

from Hardware_Control import CMT_VNA_functions as vna


def main():
    print("test hulo")
    vna.restart_app()
    print(vna.Measure_S21(80, 1, "b1_pcb", 0))
    
if __name__ == "__main__":
    main()
