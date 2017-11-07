# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
#import read_csv
import pywt  # python 小波变换的包

# data = read_csv.data
# data = np.array(data)

class wavelet:
    def __init__(self,index_list, wavefunc='db4', lv=4, m=2, n=4, plot=True):
        self.index_list = index_list
        self.wavefunc = wavefunc
        self.lv = lv
        self.m = m
        self.n = n
        self.plot = plot

    def wt(self):
        coeff = pywt.wavedec(self.index_list, self.wavefunc, mode='sym', level=self.lv)
        sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0
        # denoising
        for i in range(self.m, self.n + 1):
            cD = coeff[i]
            for j in range(len(cD)):
                Tr = np.sqrt(2 * np.log(len(cD)))
                if cD[j] >= Tr:
                    coeff[i][j] = sgn(cD[j]) - Tr
                else:
                    coeff[i][j] = 0

        denoised_index = pywt.waverec(coeff, self.wavefunc)
        if self.plot == True:
            plt.figure()
            plt.plot(self.index_list)
            plt.plot(denoised_index)
            plt.savefig('pic/denoised.png')
            plt.show()
        return denoised_index



