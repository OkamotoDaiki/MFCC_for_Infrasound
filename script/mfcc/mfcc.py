#coding:utf-8
import wave
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.fftpack.realtransforms
import copy
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('error')


class MFCCclass():
    """
    This class is to make MFCC with signal. Output 12 dimention vector.
    """
    def __init__(self, input_signal, fs, N, numChannels=20, fo=0.4, mel=1000, cutpoint=12, p_filter=0.97):
        """
        numChannels: number of filterbank
        fs: sampling rate
        input_signal:input signal data. type of list
        start: start input data
        N: number of samples, or window width
        """
        self.input_signal= input_signal
        self.fs = fs
        self.N = N
        self.numChannels = numChannels
        self.fo = fo
        self.mel = mel
        self.cutpoint = cutpoint
        self.p_filter = p_filter


    def PreEmphasisFilter(self):
        """
        Make PreEmphasisFilter using hamming window. constant p_filter = 0.97.
        """
        def get_preEmphasis(signals, p):
            """
            Make FIR[1.0, -9] filter.
            """
            return scipy.signal.lfilter([1.0,-p],1,signals)

        #p_filter: プレエンファシスフィルタ
        return get_preEmphasis(self.input_signal, self.p_filter)  


    def melFilterBank(self):
        """
        Get mel filter bank.
        """
        mo = self.mel / np.log((self.mel / self.fo) + 1)
        
        def Hz2mel(f):
            """
            transform Hz to mel.
            """
            return mo * np.log(f / self.fo + 1.0)
        

        def mel2hz(m):
            """
            transform mel to Hz
            """
            return self.fo * (np.exp(m / mo) - 1.0)

        fmax = self.fs / 2 #ナイキスト周波数
        melmax = Hz2mel(fmax) #ナイキスト周波数(mel)
        nmax = int(self.N / 2) #周波数インデックスの最大数
        df = self.fs / self.N #周波数解像度

        #メル尺度における各フィルタの中心周波数を求める
        #dmel = melmax / (self.numChannels + 1)
        dmel = melmax / (self.numChannels+1)
        melcenters = np.arange(1, self.numChannels + 1) * dmel
        #各フィルタの中心周波数をHzに変換
        fcenters = mel2hz(melcenters)

        #各フィルタの中心周波数を周波数インデックスに変換
        indexcenter = np.round(fcenters / df)
        
        #各フィルタの開始位置のインデックス
        indexstart = np.hstack(([0], indexcenter[0:self.numChannels - 1]))

        # 各フィルタの終了位置のインデックス
        indexstop = np.hstack((indexcenter[1:self.numChannels], [nmax]))

        filterbank = np.zeros((self.numChannels, nmax))
        for c in np.arange(0,self.numChannels):
            #三角フィルタの左の直線の傾きから点を求める
            increment = 1.0 / (indexcenter[c] - indexstart[c])
            for i in np.arange(indexstart[c], indexcenter[c]):
                filterbank[int(c), int(i)] = (i - indexstart[c]) * increment
            #三角フィルタの右の直線の傾きから点を求める
            decrement = 1.0 / (indexstop[c] - indexcenter[c])
            for i in np.arange(indexcenter[c], indexstop[c]):
                filterbank[int(c),int(i)] = 1.0 - ((i - indexcenter[c]) * decrement)
        
        return filterbank, fcenters


    def find_cutpoint(self, freq_seq, nq_fft_list):
        """
        Find Cutpoint for highpass-filter.
        """
        magnitude = 1
        count = 0
        min_freq = 0
        min_array_number = 0

        threshold_list = []
        for i in range(len(nq_fft_list)):
            mean_fft_amp = np.mean(nq_fft_list)
            var_fft_amp = np.std(nq_fft_list)
            threshold = mean_fft_amp + magnitude * var_fft_amp
            analysis_freq = nq_fft_list[i]
            if analysis_freq >= threshold:
                nq_fft_list[i] = 0
                count += 1
            else:
                min_freq = freq_seq[i + 1]
                min_array_number = count + 1
                #print("min frequency = {}".format(min_freq))
                break    
            threshold_list.append(threshold)
        return min_freq, min_array_number, threshold_list


    def highpassfilter(self, freq_seq, fft_data, dF):
        """
        Highpass Filter
        """
        fc = dF * self.cutpoint
        fft_highpass = copy.copy(fft_data)

        count = 0
        for freq in freq_seq:
            if freq - fc >=0:
                break
            count += 1
        fft_highpass[:count] = 0
        fft_highpass[len(fft_highpass) - count:] = 0
        return fft_highpass


    def mfcc(self):
        """
        Get MFCC from transfroming spectrum.
        Cutpoint have a role cutting mel filter bank. Defalut is 12.
        """
        def get_mean(array):
            return sum(array) / len(array)

        def assign_mean_to_zero(array):
            """
            If antilogarithm is 0, assign mean with other data
            """
            zero_array_numbers = [i for i in range(len(array)) if array[i] == 0]
            not_zero_array_numbers = [i for i in range(len(array)) if array[i] != 0]
            mean = get_mean(not_zero_array_numbers)
            for i in zero_array_numbers:
                array[i] = mean
            return array


        dF = self.fs / self.N
        PEf_Data = self.PreEmphasisFilter()
        windowedData = np.hamming(self.N) * PEf_Data
        fft_data = abs(np.fft.fft(windowedData))[:int(self.N/2)]
        #higpass-filter        
        #freq_seq = np.arange(0, self.fs, dF)
        #nq_fft_list = abs(fft_data)[:int(self.N / 2)]
        #min_freq, min_array_number, threshold_list = self.find_cutpoint(freq_seq, nq_fft_list)
        #fft_highpass = self.highpassfilter(freq_seq, fft_data, dF, min_array_number)

        filterbank, fcenters = self.melFilterBank() #フィルタバンクを求める
        inner_product_fbank = np.dot(fft_data, filterbank.T)
        modify_dot = assign_mean_to_zero(inner_product_fbank)
        mspec = np.log10(modify_dot) #スペクトル領域にフィルタバンクをかける
        ceps = scipy.fftpack.realtransforms.dct(mspec,type=2,norm="ortho",axis=-1) #離散コサイン変換
        return ceps[1:self.cutpoint+1]


    def get_melfilterbank(self, fname):
        """
        Generate Mel-filterbank graph.
        """
        #メルフィルタバンク
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.clf()
        filterbank, fcenters = self.melFilterBank()
        #周波数解像度
        df = self.fs / self.N
        for i in np.arange(0, self.numChannels):
            #plt.plot(np.arange(0, self.N)* df, filterbank[i])
            plt.plot(np.arange(0,len(filterbank[i]))*df, filterbank[i])
        plt.xlim(0, self.fs / 2)
        plt.ylim(0, 1.0)
        plt.xlabel("frequency")
        plt.savefig(fname + '_melfilterbank.png')
        return 0


def delta_cepstrum(mfcc_list, cutpoint=12):
    """
    Calculate delta-cepstrum.
    """
    delta_cepstrum = []
    for i in range(cutpoint):
        cov = 0
        k_std = 0
        for kframe in range(len(mfcc_list)):
            cov = cov + (kframe * mfcc_list[kframe][i])
            k_std = k_std + kframe ** 2
        delta_c = cov / k_std
        delta_cepstrum.append(delta_c)
    return delta_cepstrum