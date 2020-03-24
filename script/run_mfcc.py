import numpy as np
import pandas as pd
import sys
import shutil
import os
import glob
import pickle
#my module
from mfcc import mfcc
from subscript import OperateFpath

#freq parameter
fo = 0.4
nframe = 256


def WritePickle(object_data, fpath, fname):
    """
    Save binary data with pickle module.
    
    """
    with open(fpath, 'wb') as f:
        pickle.dump(object_data, f)
    #print("Save '{}'".format(fpath))
    return 0


def TransformMLFormat(count_label, *args):
    """
    mfcc ML data format : save pkl and [label, mfcc_data, delta_ceps]
    """
    feature = []
    for arg in args:
        feature = feature + list(arg)
    #print(feature)
    #print("feature dim = {}".format(len(feature)))
    return [count_label, feature]


def GetMFCC(data, fs=2, numChannels=20):
    """
    Get MFCC from mfcc script
    """
    numChannels = 20
    N = len(data)
    #print(N)
    MFCCclass_obj = mfcc.MFCCclass(data,fs,N,numChannels)
    mfcc_data = MFCCclass_obj.MFCC(fo=fo)
    #print(mfcc_data)
    return mfcc_data


def SepFrame(data, nframe=nframe, ov=0.75):
    """
    Separate data with nframe length including overlap.
    number of frame is, int(1 + ((N/ nframe) - 1) / (1 - ov))
    """
    def RestractOverlapValue(ov):
        if ov < 0 or ov > 1:
            print("overlap rate have to set 0 < ov < 1.")
            sys.exit()
        return 0

    def GetStartWindow(shift, N):
        """
        Get numbers array of window for overlap
        """
        start_window = []
        i = 0
        while shift * i <= N:
            start_window.append(int(shift * i))
            i+=1
        return start_window

    def GetCombforWindow(start_window, nframe):
        """
        Get combinations window both ends.
        """
        #print("length of start_window = {}".format(len(start_window)))
        combs = []
        #num_array_first_nframe = start_window.index(nframe)

        for i in range(len(start_window)):
            if start_window[i] + nframe <= start_window[-1]:
                comb = (start_window[i], start_window[i] + nframe)
                combs.append(comb)
            else:
                break
        #print(combs)
        return combs
    
    def GetWindowData(combs, data):
        """
        Get window data with output from GetCombforWindow function
        """
        all_sep_data = []
        for comb in combs:
            sep_data = []
            start = comb[0]
            end = comb[1]
            for i in range(len(data)):
                if i >= start and i < end:
                    sep_data.append(data[i])
                if i >= end or i == len(data)-1:
                    all_sep_data.append(sep_data)
                    break
        return all_sep_data

    #debug
    RestractOverlapValue(ov)

    #edit now
    N = len(data)
    shift = nframe * (1 - ov)
    
    start_window = GetStartWindow(shift, N)
    combs = GetCombforWindow(start_window, nframe)
    all_sep_data = GetWindowData(combs, data)
    #edit end
    return all_sep_data


def DeltaCeps(data):
    """
    Flow calculating delta cepstrum. The mfcc module does the calculation of delta cepstrum.
    bug:
    """
    #delta cepstrum
    #print("delta cepstrum...")
    sep_data = SepFrame(data, nframe=nframe)
    mfcc_list = [GetMFCC(data) for data in sep_data]
    cutpoint = list(set([len(mfcc_data) for mfcc_data in mfcc_list]))[0]
    delta_cepstrum = mfcc.DeltaCepstrum(mfcc_list, cutpoint=cutpoint)
    #print(delta_cepstrum)
    return delta_cepstrum


def ChoiceFeature(feature_mode, label, data, fs):
    """
    Adjust choice feature.
    """
    if feature_mode == 'mfcc_and_delta-ceps':
        mfcc_data = GetMFCC(data, fs=fs)
        delta_ceps = DeltaCeps(data)
        ML_format_data = TransformMLFormat(label, mfcc_data, delta_ceps)
    elif feature_mode == 'mfcc':
        mfcc_data = GetMFCC(data, fs=fs)
        ML_format_data = TransformMLFormat(label, mfcc_data)
    elif feature_mode == 'delta-ceps':
        delta_ceps = DeltaCeps(data)
        ML_format_data = TransformMLFormat(label, delta_ceps)
    else:
        print("Error : Wrong inputing feature_mode. Modify script")
        sys.exit()
    return ML_format_data


def ModifyFileStructure(feature_mode_list, threshold_variable_list, pkl_folder_fpath, label_list):
    """
    Modify File Structure for input machine learning.
    Because of being different from GSC file structure. This method is to the same structure of GSC directory.
    """
    for feature_mode in feature_mode_list:
        feature_fpath = pkl_folder_fpath + "/" + feature_mode
        os.mkdir(feature_fpath)
        for threshold_variable in threshold_variable_list:
            pkl_threshold_fpath = feature_fpath + "/" + threshold_variable
            os.mkdir(pkl_threshold_fpath)

            #modify file structure for input machine learning.
            filepaths = {
                "label_signal" : "../supervise_data" + "/" + threshold_variable + "/" + "supervise_label_0/"
            }

            for key_value in filepaths.items():           
                folder_names = OperateFpath.GetAllMultiFolder(key_value[1])
                for target_obs in folder_names:
                    pkl_target_fpath =  pkl_threshold_fpath + "/" + target_obs
                    os.mkdir(pkl_target_fpath)
    return 0


def ReadSeqData(csv_fpath):
    """
    Read preprocessed infrasound data.
    """
    df = pd.read_csv(csv_fpath)
    times = df["SensorTimeStamp"].tolist()
    data = df["InfAC"].tolist()
    if len(times) == 0 and len(data) == 0:
        print("Error: data is nothing.")
    return times, data


def GetMLobject(label_list, threshold_variable, place_name, fs=2):
    """
    Generate object of transforming to MFCC.
    """
    write_ML_data = []

    filepaths = {
        "label_signal" : "../supervise_data" + "/" + threshold_variable + "/" + "supervise_label_0",
        "label_noise" : "../supervise_data" + "/" + threshold_variable + "/" + "supervise_label_1"
    }

    for key_value in filepaths.items():
        label_kind = key_value[0]
        label_fpath = key_value[1]
        if label_kind == label_list[0]:
            label = 1
        elif label_kind == label_list[1]:
            label = 0
        else:
            print("Error : Missing input label. Modify script.")
            sys.exit()
        
        place_fpath = label_fpath + "/" + place_name
        csv_fpaths = glob.glob(place_fpath + "/*.csv")
        for csv_fpath in csv_fpaths:
            times, data = ReadSeqData(csv_fpath)
            required_data = [label, fs, data]
            write_ML_data.append(required_data)
    return write_ML_data


def main():
    #init
    fs = 2
    label_number = 0
    fs_number = 1
    data_number = 2
    feature_mode_list = ["mfcc_and_delta-ceps", "mfcc", "delta-ceps"]
    label_list = ["label_signal", "label_noise"]
    supervise_data_fpath = "../supervise_data/"
    place_name_fpath = "../supervise_data/0div10mag/supervise_label_1"

    #Generate saving pkl file
    pkl_folder_fpath = "../" + "pkl_file"
    try:
        shutil.rmtree(pkl_folder_fpath)
        os.mkdir(pkl_folder_fpath)
    except FileNotFoundError:
        os.mkdir(pkl_folder_fpath)

    #Get threshold variable directory
    threshold_variable_list = OperateFpath.GetAllMultiFolder(supervise_data_fpath)
    place_name_list = OperateFpath.GetAllMultiFolder(place_name_fpath)
    print("Generate folder for pkl file.")
    ModifyFileStructure(feature_mode_list, threshold_variable_list, pkl_folder_fpath, label_list)

    #read csv data.
    for feature_mode in feature_mode_list:
        for threshold_variable in threshold_variable_list:
            for place_name in place_name_list:
                zero_div_count = 0
                save_fpath = pkl_folder_fpath + "/" + feature_mode + "/" + threshold_variable + "/" + place_name
                print("save fpath = \n{}".format(save_fpath))
                print("Generate object data for transform feature...")
                object_data = GetMLobject(label_list, threshold_variable, place_name, fs=fs)
                data_length = len(object_data)
                #write pkl
                pkl_file_number = 0
                for i in range(data_length):
                    label = object_data[i][label_number]
                    fs = object_data[i][fs_number]
                    cut_data = object_data[i][data_number]
                    try:
                        #print("Generate feature vector..")
                        ML_format_data = ChoiceFeature(feature_mode, label, cut_data, fs)
                        save_fname = save_fpath + "/" + "label_" + str(label) + "_" + str(i) + "_" + feature_mode + ".pkl"
                        WritePickle(ML_format_data, save_fname, save_fname)
                        pkl_file_number += 1
                    except ZeroDivisionError:
                        zero_div_count += 1
                        print("Error: zero division calculating mean.")
                print("Nmber of generating pkl file = {}".format(pkl_file_number))
                print("Zero division count = {}".format(zero_div_count))
    print("finish")
    return 0


if __name__ == '__main__':
    main()