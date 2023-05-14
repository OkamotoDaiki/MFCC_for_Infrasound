import numpy as np
import pandas as pd
import sys
import shutil
import os
import glob
import pickle
import json
#my module
from mfcc import mfcc
from subscript import OperateFpath

def write_pickle(object_data, fpath, fname):
    """
    Save binary data with pickle module.
    
    """
    with open(fpath, 'wb') as f:
        pickle.dump(object_data, f)
    return 0


def transform_ML_format(count_label, *args):
    """
    mfcc ML data format : save pkl and [label, mfcc_data, delta_ceps]
    """
    feature = []
    for arg in args:
        feature = feature + list(arg)
    return [count_label, feature]


def get_mfcc(data, fs, numChannels, cutpoint, fo, mel):
    """
    Get MFCC from mfcc script
    """
    N = len(data)
    MFCCclass_obj = mfcc.MFCCclass(data, fs, N, numChannels, cutpoint, fo, mel)
    mfcc_data = MFCCclass_obj.MFCC()
    return mfcc_data


def separate_frame(data, nframe, ov):
    """
    Separate data with nframe length including overlap.
    number of frame is, int(1 + ((N/ nframe) - 1) / (1 - ov))
    """
    def restart_overlap_value(ov):
        if ov < 0 or ov > 1:
            print("overlap rate have to set 0 < ov < 1.")
            sys.exit()
        return 0

    def get_start_window(shift, N):
        """
        Get numbers array of window for overlap
        """
        start_window = []
        i = 0
        while shift * i <= N:
            start_window.append(int(shift * i))
            i+=1
        return start_window

    def get_comb_for_window(start_window, nframe):
        """
        Get combinations window both ends.
        """
        combs = []

        for i in range(len(start_window)):
            if start_window[i] + nframe <= start_window[-1]:
                comb = (start_window[i], start_window[i] + nframe)
                combs.append(comb)
            else:
                break
        return combs
    
    def get_window_data(combs, data):
        """
        Get window data with output from get_comb_for_window function
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
    restart_overlap_value(ov)

    #edit now
    N = len(data)
    shift = nframe * (1 - ov)
    
    start_window = get_start_window(shift, N)
    combs = get_comb_for_window(start_window, nframe)
    all_sep_data = get_window_data(combs, data)
    #edit end
    return all_sep_data


def get_delta_ceps(data, fs, numChannels, cutpoint, fo, mel, nframe, ov):
    """
    Flow calculating delta cepstrum. The mfcc module does the calculation of delta cepstrum.
    bug:
    """
    #delta cepstrum
    sep_data = separate_frame(data, nframe, ov)
    mfcc_list = [get_mfcc(data, fs, numChannels, cutpoint, fo, mel) for data in sep_data]
    cutpoint = list(set([len(mfcc_data) for mfcc_data in mfcc_list]))[0]
    delta_cepstrum = mfcc.DeltaCepstrum(mfcc_list, cutpoint=cutpoint)
    return delta_cepstrum


def choose_feature(feature_mode, label, data, fs, numChannels, cutpoint, fo, mel, nframe, ov):
    """
    Adjust choice feature.
    """
    if feature_mode == 'mfcc_and_delta-ceps':
        mfcc_data = get_mfcc(data, fs, numChannels, cutpoint, fo, mel)
        delta_ceps = get_delta_ceps(data, fs, numChannels, cutpoint, fo, mel, nframe, ov)
        ML_format_data = transform_ML_format(label, mfcc_data, delta_ceps)
    elif feature_mode == 'mfcc':
        mfcc_data = get_mfcc(data, fs, numChannels, cutpoint, fo, mel)
        ML_format_data = transform_ML_format(label, mfcc_data)
    elif feature_mode == 'delta-ceps':
        delta_ceps = get_delta_ceps(data, fs, numChannels, cutpoint, fo, mel, nframe, ov)
        ML_format_data = transform_ML_format(label, delta_ceps)
    else:
        print("Error : Wrong inputing feature_mode. Modify script")
        sys.exit()
    return ML_format_data


def modify_file_structure(feature_mode_list, threshold_variable_list, pkl_folder_fpath, label_list):
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


def read_preprocessed_data(csv_fpath):
    """
    Read preprocessed infrasound data.
    """
    df = pd.read_csv(csv_fpath)
    times = df["SensorTimeStamp"].tolist()
    data = df["InfAC"].tolist()
    if len(times) == 0 and len(data) == 0:
        print("Error: data is nothing.")
    return times, data


def get_ML_object(label_list, threshold_variable, place_name, fs=2):
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
            times, data = read_preprocessed_data(csv_fpath)
            required_data = [label, fs, data]
            write_ML_data.append(required_data)
    return write_ML_data


def main():
    # JSONファイルを読み込む
    with open('./config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    #init
    """
    object_dataの構造
    object_data = [教師ラベル, サンプリングレート, データ列]
    """
    _label_number = 0 #object_dataのラベルが書かれている要素番号
    _fs_number = 1 #object_dataのサンプリングレートが書かれている要素番号
    _data_number = 2 #object_dataのデータが書かれている要素番号
    feature_mode_list = ["mfcc_and_delta-ceps", "mfcc", "delta-ceps"] #生成する特徴量の種類
    label_list = [config["label_1"], config["label_0"]] #ラベルの判定要素
    supervise_data_fpath = config["supervise_data_fpath"] #教師データのフォルダパス
    place_name_fpath = config["place_name_fpath"] #観測点を抽出するための初期値
    fs = config["fs"] #サンプリングレート
    numChannels = config["numChannels"] #MFCCのチャネル数
    cutpoint = config["cutpoint"] #MFCCのチャネル数のカットポイント
    fo = config["fo"] #MFCCの周波数パラメータ
    mel = config["mel"] #MFCCにおけるメル尺度パラメータ
    ov = config["ov"] #オーバーラップ率
    nframe = config["nframe"] #窓幅

    #Generate saving pkl file
    pkl_folder_fpath = config["pkl_folder_fpath"] #出力ファイルのフォルダパス
    try:
        shutil.rmtree(pkl_folder_fpath)
        os.mkdir(pkl_folder_fpath)
    except FileNotFoundError:
        os.mkdir(pkl_folder_fpath)

    #Get threshold variable directory
    threshold_variable_list = OperateFpath.GetAllMultiFolder(supervise_data_fpath)
    place_name_list = OperateFpath.GetAllMultiFolder(place_name_fpath)
    print("Generate folder for pkl file.")
    modify_file_structure(feature_mode_list, threshold_variable_list, pkl_folder_fpath, label_list)

    #read csv data.
    for feature_mode in feature_mode_list:
        for threshold_variable in threshold_variable_list:
            for place_name in place_name_list:
                zero_div_count = 0
                save_fpath = pkl_folder_fpath + "/" + feature_mode + "/" + threshold_variable + "/" + place_name
                print("save fpath = ".format(save_fpath))
                print("Generate object data for transform feature...")
                object_data = get_ML_object(label_list, threshold_variable, place_name, fs=fs)
                data_length = len(object_data)
                #write pkl
                pkl_file_number = 0
                for i in range(data_length):
                    label = object_data[i][_label_number]
                    fs = object_data[i][_fs_number]
                    cut_data = object_data[i][_data_number]
                    try:
                        ML_format_data = choose_feature(feature_mode, label, cut_data, fs, numChannels, cutpoint, fo, mel, nframe, ov)
                        save_fname = save_fpath + "/" + "label_" + str(label) + "_" + str(i) + "_" + feature_mode + ".pkl"
                        write_pickle(ML_format_data, save_fname, save_fname)
                        pkl_file_number += 1
                    except ZeroDivisionError:
                        zero_div_count += 1
                        print("Error: zero division calculating mean.")
                print("Number of generating pkl file = {}".format(pkl_file_number))
                print("Zero division count = {}".format(zero_div_count))
    print("finish")
    return 0


if __name__ == '__main__':
    main()