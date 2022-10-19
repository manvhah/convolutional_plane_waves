"""
Data management, reading from robot measurements to hdf, loading hdf and so on.

Ideas, to be added / changed
- impulse response
- store sampling rate Fs in a variable
- description & units for measurements in header
"""

import numpy as np
import os
import h5py as hdf


def read_hdf(filename, groups = None):
    """
    read data from hdf file
   
    params:
        filename
        groups (default: None) select a specific group

    returns:
        dictionary holding the data
    """
    if not filename[-3:] == '.h5':
        filename += '.h5'
    data = dict()

    with hdf.File(filename, 'r') as f:
        if groups == None: #no group specified, read all
            groups = f.keys()
        for g in groups:
            # print(f[g])
            data.update({ g : dict() })
            for k in f[g].keys():
                data[g].update({ k : np.array(f[g][k]) })
    return data


def write_hdf(data_dict, filename = './data.h5'):
    """ stores dictionary in hdf format
    parameters:
        data_dict   dictionary holding the data
        filename    file name to be stored (default: ./data.h5)
    """
    if not filename[-3:] == '.h5':
        filename += '.h5'

    if os.path.isfile(filename):
        print("! modifying existing file {} !".format(filename))

    with hdf.File(filename, "a") as f:
        group_name = '_'.join(data_dict['rawfile'][0].split('/')[:-1])
        if group_name in f.keys():
            # print("! overwriting group {} !".format(group_name))
            del f[group_name]
        group = f.create_group(group_name)

        data_dict.pop('rawfile',None) # filter rawfile field
        for att in data_dict.keys():
            group.create_dataset(att, data = data_dict[att])

    return filename


def read_measurements(path, filename, remote_dir, test_flag = False):
    """ reads raw measurements, sotres them in hdf and returns a dictionary with
    the full data
    """
    for root, dirs, files in os.walk(path):
        print("\nChecking files in {}".format(root))
        # load and store each measurement series
        if len(_gen_local_data_list(root)): # skip dir if no data is found
            series = _load_mat_series(root, remote_dir, test_flag)
            write_hdf(series, filename)
            # read complete data from the file
    data = read_hdf(filename)
    return data


def _gen_local_data_list(root, ending = ".mat"):
    """
    generate list of absolute paths to all files with given ending.
    params:
        basepath : directory where to search for files
        ending : string with last characters of filename. default ".mat"
    returns:
        file_list : list of absolute filepaths
    """
    flist = list()
    for file in os.listdir(root):
        if file.endswith('.mat'):
            flist.append(os.path.join(root, file))
    flist = [ff for ff in flist if 'Response' in ff]
    return flist


def _load_mat_series(directory, remote_dir = None, trunc_flag = False):
    """
    load all data from a list of files in one directory and return dictionary with lists and
    arrays respectively
   
    truncates at x Hz
    if available, translates with coordinates stored in $(directory)/PosRobot.mat
    """
    import scipy.io as sio
    import re
    import time
    fpath_list = _gen_local_data_list(directory, ending='.mat')

    if trunc_flag: # for testing
        fpath_list = fpath_list[:20]

    ## look for robot position and store in translation rel -> abs coords
    translation = np.zeros(3)
    config_file_path =''.join([directory,'/measurement_configuration.mat'])
    # if os.path.isfile(config_file_path): # CONFIG FILE NEEEDED NOW
    config_data = sio.loadmat(config_file_path)
    translation = config_data['PosRobot'][0]
    translation[(translation%10 == translation)] *=1000 # scaling to mm

    # translation[2] = 1000 # data is rel robot center at 1m height
    print('''translate positions by {} {} {} mm, found in: {}'''.
        format(*translation, '/'.join(config_file_path.split('/')[-3:])))

    ## read data and move to absolute coordinates
    # flist     = np.arange(0,5001)*10 # previous, 1 Hz resolution up to 5k
    flist     = np.arange(0,50001) # full .1 Hz up to 5k
    flen      = flist.shape[0]
    lenlist   = len(fpath_list)
    timestamp = list()
    fname     = list()
    measurement_id = np.empty(lenlist).astype(np.int)
    position  = np.empty((lenlist,3))
    response  = np.empty((flen,lenlist)).astype(np.complex128)

    for ii,file_path in enumerate(fpath_list):
        # print('... {}'.format(os.path.basename(file_path)))
        try:
            dd = sio.loadmat(file_path)
        except:
            print('! error loading local {}, try remote'.format(os.path.basename(file_path)))
            if os.path.exists(remote_dir):
                try:
                    remote_file = ''.join([remote_dir,'/',os.path.basename(file_path)])
                    dd = sio.loadmat(remote_file)
                    try: # try update local file if permissions are ok
                        from shutil import copyfile
                        copyfile(remote_file, file_path)
                    except:
                        print('! not downloading {}'.format(os.path.basename(file_path)))
                except:
                    print('! error loading remote , skipping {}'.format(os.path.basename(file_path)))
                    continue
            else:
                print('! remote not found, skipping {}'.format(os.path.basename(file_path)))
                continue

        fname.append('/'.join(file_path.split('/')[-3:]))
        response[:,ii] = dd['FR'].flatten()[flist]
        timestamp.append(
                time.strptime(str(dd['__header__']).split(',')[-1][13:-1]) )
        pos_idx = int(re.findall(r'\d+\b', file_path)[-1]) - 1
        measurement_id[ii] = pos_idx + 1
        position[ii,:] = config_data['PosMic'][pos_idx,:]

    frequency = config_data['f'].flatten()[flist]

    position[(position%10 == position)] *=1000
    position[:,:2] = np.round(position[:,:2]/5)*5 # round to 5 mm
    transformation = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    position = position@transformation + translation

    data_dict = dict({'rawfile'  : fname,
                'timestamp' : timestamp,
                'xyz'       : position.astype(np.int),
                'frequency' : frequency,
                'response'  : response,
                'id'        : measurement_id,
                })
           
    try:
        temp_file_path =''.join([directory,'/hist_temperature.mat'])
        temp_data   = sio.loadmat(temp_file_path)
        temp_values = temp_data['temperature'][:,0]
        temp_time   = [_time_struct_from_ordinal(tt) for tt in temp_data['datetime']]

        timestamp_vec = np.array([time.mktime(tt) for tt in timestamp])
        temp_time_vec = np.array([time.mktime(tt) for tt in temp_time])
       
        # piecewise linear interpolation
        temperature = np.interp(timestamp_vec, temp_time_vec, temp_values)

        data_dict.update({'outside_temperature': temperature})
    except:
        print(' no temp data found , continuing... ')

    return data_dict

from datetime import datetime, timedelta
def _time_struct_from_ordinal(x, tz=None):
    ix = int(x)
    dt = datetime.fromordinal(ix)
    remainder = float(x) - ix
    hour, remainder = divmod(24 * remainder, 1)
    minute, remainder = divmod(60 * remainder, 1)
    second, remainder = divmod(60 * remainder, 1)
    microsecond = int(1e6 * remainder)
    if microsecond < 10:
        microsecond = 0  # compensate for rounding errors
    dt = datetime(dt.year-1, dt.month, dt.day, int(hour), int(minute),
                  int(second), microsecond)
    if tz is not None:
        dt = dt.astimezone(tz)

    if microsecond > 999990:  # compensate for rounding errors
        dt += timedelta(microseconds=1e6 - microsecond)

    return dt.timetuple()


def read_all_measurements():

    ### use test_flag to only load twenty measurements per folder

    # path = '/home/manu/dl/Measurements/room_011_flanking_transmission'
    # target_name = 'room_011'

    path = '/home/manu/dl/Measurements/room_019_lecture_room/P3'
    remote_path = '/mnt/act/act/arch/room 019 extensive measurements/Measurements/P3'
    target_name = '019_lecture_room_p3'
    # target_name = '019_lecture_room_full'
   
    ## read all data from path, takes a long time
    data = read_measurements(path,
            remote_dir = remote_path,
            test_flag  = False,
            filename   = target_name)

    # path = '/home/manu/dl/Measurements/room_019_lecture_room/P1'
    # remote_path = '/mnt/act/act/arch/room 019 extensive measurements/Measurements/P1'
   
    # ## read all data from path, takes a long time
    # data = read_measurements(path,
            # remote_dir = remote_path,
            # test_flag  = False,
            # filename   = target_name)

    # path = '/home/manu/dl/Measurements/room_019_lecture_room/P2'
    # remote_path = '/mnt/act/act/arch/room 019 extensive measurements/Measurements/P2'
   
    # ## read all data from path, takes a long time
    # data = read_measurements(path,
            # remote_dir = remote_path,
            # test_flag  = False,
            # filename   = target_name)

## plot positions of data

    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.clf()
    # tdata = read_hdf('019_lecture_room.h5')
    # lim = [-1000,4e3]
    # # rlim = np.array([[0,4.41],[0,3.31],[0,2.97]])*1000
    # rlim = np.array([[0,6.63],[0,9.45],[0,2.97]])*1000
    # for k in tdata.keys():
        # xyz = tdata[k]['xyz']
        # print(k)
        # print(xyz[0,:])
        # plt.subplot(311), plt.plot(xyz[:,0],xyz[:,1],'.', alpha=.5)
        # plt.xlim(rlim[0,:]),
        # plt.ylim(rlim[1,:])
        # plt.axis('equal')
        # plt.subplot(312), plt.plot(xyz[:,0],xyz[:,2],'.', alpha=.5)
        # plt.xlim(rlim[0,:]),
        # plt.ylim(rlim[2,:])
        # plt.axis('equal')
        # plt.subplot(313), plt.plot(xyz[:,1],xyz[:,2],'.', alpha=.5)
        # plt.xlim(rlim[1,:]),
        # plt.ylim(rlim[2,:])
        # plt.axis('equal')
    # plt.draw()
    # plt.show()

