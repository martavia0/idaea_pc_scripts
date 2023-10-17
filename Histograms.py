# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:09:26 2023

@author: Marta Via
"""

import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import numpy as np
import scipy
import glob
import math

class Histogram_treatment:
    
    def __init__(self,x):
        self.x = 0
        pass
    
    def Hello(self):
        print("Hello Histogram player")
                
    def histogram_intersection_integral(self, h1, h2, nbins, rang):
        sm = 0
        sM=0
        dx=rang*2.0/nbins #dx és le quantitat de x que té cada bin, rang de (0, rang)*2 (per tenir positiu i negatiu) / nbins
        for i in range(nbins):
            sm += min(h1[i], h2[i])*dx #Integral discreta
        return sm*100/2.0
    
    def histogram_intersection_basic(self, h1, h2, bins):
        sm = 0
        sM=0
        for i in range(bins):
            sm = sm + min(h1[i], h2[i])
            sM = sM + max(h1[i], h2[i])
        return sm*100/sM
    
    def histogram_intersection_trapezoids(self, h1, h2, bins):
    
        min_f = [min(h1[i], h2[i]) for i in range(0,len(h1))]
        max_f = [max(h1[i], h2[i]) for i in range(0,len(h2))]
        hist_int = sp.trapezoid(min_f, x=bins[:-1])*100.0/np.trapz(max_f, x=bins[:-1])
        return hist_int