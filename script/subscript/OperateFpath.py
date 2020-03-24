import os
import glob

def MultiGetSavePathandTime(fpath, vol_place, obs_place):
    """
    Get save path and JMA got observation time only one.
    """
    generate_foldername = "/graph"
    get_fcsv_data = "/interpolation_data"

    filenames = os.listdir(fpath)
    files_dir = [f for f in filenames if os.path.isdir(os.path.join(fpath, f))]

    Investigate_folder = [fname for fname in files_dir \
        if vol_place in fname and obs_place in fname]

    obs_time_JMA_list = []
    get_fcsv_data_list = []
    generate_graph_fpath_list = []

    for folder in Investigate_folder:
        #JMA time format
        date, time = GetObsTimeJMA(folder)
        obs_time_JMA = TimeFormat(date, time)
        obs_time_JMA_list.append(obs_time_JMA)
        #csv file path
        get_fcsv_data_fpath = fpath + "/" + folder + get_fcsv_data
        get_fcsv_data_list.append(get_fcsv_data_fpath)
        #graph path
        generate_graph_fpath = fpath + "/" + folder + generate_foldername
        generate_graph_fpath_list.append(generate_graph_fpath)
    return obs_time_JMA, get_fcsv_data_fpath, generate_graph_fpath


def GetObsPlaceName(csv_file, infs_number, lack_rate_number):
    """
    Get infs observatory place Name
    """
    obs_place_name_list = csv_file.split("/")[-1].split("_")[infs_number:lack_rate_number]
    print(obs_place_name_list)
    obs_place = ""
    num = 0
    for sub_obs_place in obs_place_name_list:
        num += 1
        if num == len(obs_place_name_list):
            obs_place = obs_place + sub_obs_place
        else:
            obs_place = obs_place + sub_obs_place + "_"    
    return obs_place


def GetMultiFolder(fpath, vol_place, obs_place):
    """
    Get specified multi folder.
    """
    filenames = os.listdir(fpath)
    files_dir = [f for f in filenames if os.path.isdir(os.path.join(fpath, f))]

    Investigate_folder = [fname for fname in files_dir \
        if vol_place in fname and obs_place in fname]    
    return Investigate_folder


def GetAllMultiFolder(fpath):
    """
    Get specified multi folder.
    """
    filenames = os.listdir(fpath)
    files_dir = [f for f in filenames if os.path.isdir(os.path.join(fpath, f))]   
    return files_dir


def SingleGetSavePathandTime(fpath, folder):
    """
    Get save path and JMA got observation time only one.
    """
    generate_foldername = "/graph"
    get_fcsv_data = "/interpolation_data"

    #JMA time format
    date, time = GetObsTimeJMA(folder)
    obs_time_JMA = TimeFormat(date, time)
    #csv file path
    get_fcsv_data_fpath = fpath + "/" + folder + get_fcsv_data
    #graph path
    generate_graph_fpath = fpath + "/" + folder + generate_foldername
    return obs_time_JMA, get_fcsv_data_fpath, generate_graph_fpath


def TimeFormat(date, time):
    """
    TimeFormat yyyymmdd hhmm -> yyyy-mm-dd hh:mm:00
    """
    year = date[:4]
    month = date[4:6]
    day = date[6:8]

    hour = time[:2]
    minute = time[2:4]
    timeformat = year + "-" + month + "-" + day + " " + hour + ":" + minute + ":" + "00"
    return timeformat


def GetObsTimeJMA(folder_name):
    """
    Pick up obs time JMA from folder name.
    """
    date = folder_name.split("_")[3]
    time = folder_name.split("_")[4]
    return date, time


def CSVdataPath_NewGraphFolder(csv_fpath, graph_fpath):
    """
    Make multi directory if there are multi csv file.
    """
    csv_file_list = glob.glob(csv_fpath + "/*.csv")

    new_graph_folder_name_list = []
    for csv_file_path in csv_file_list:
        new_graph_folder_name = graph_fpath + "/" + csv_file_path.split("/")[-1].split(".")[0]
        try:
            os.mkdir(new_graph_folder_name)
        except FileExistsError:
            print("already directory exist!")
        print("Make new directory to {}".format(new_graph_folder_name))
        new_graph_folder_name_list.append(new_graph_folder_name)
    return csv_file_list, new_graph_folder_name_list


def main():
    fpath = "../Infs"
    vol_place = "Sakurazima_Ontake"
    JMA_obs_place = "Higashikorimoto"

    folder_name = "Sakurazima_Ontake_Higashikorimoto_20170428_1929_1point0Pa"

    #obs_time_JMA_list, csv_fpath_list, graph_fpath_list = MultiGetSavePathandTime(fpath, vol_place, obs_place)

    obs_time_JMA, csv_fpath, graph_fpath = SingleGetSavePathandTime(fpath, folder_name)
    csv_file_list, new_graph_folder_name_list = CSVdataPath_NewGraphFolder(csv_fpath, graph_fpath)
    for folder in new_graph_folder_name_list:
        obs_place = folder.split("_")[-3] + "_" + folder.split("_")[-2]
    for csv_file in csv_file_list:
        obs_place = csv_file.split("_")[-3] + "_" + csv_file.split("_")[-2]
        print(obs_place)

    return 0

if __name__=="__main__":
    main()