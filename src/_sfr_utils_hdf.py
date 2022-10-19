import numpy as np
import h5py as hdf
import warnings

### (re-)storing from or to files
def _to_dict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            data[k] = _to_dict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return _to_dict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [_to_dict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, _to_dict(value, classkey))
            for key, value in obj.__dict__.items()
            if not callable(value) and not key.startswith('_')])
        if classkey != None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj

def _h5repack(filename):
    import subprocess
    subprocess.run("mv "+filename+" tmp.h5",shell=True)
    subprocess.run("h5repack tmp.h5 " + filename,shell=True)
    if os.path.isfile(filename):
        subprocess.run("rm tmp.h5",shell=True)
    else:
        subprocess.run("mv tmp.h5 "+filename,shell=True)


def write_to_hdf(data_dict, group_name, filename = './autosave'):
    """ stores dictionary in hdf format
    parameters:
        data_dict   class or dictionary holding the data
        filename    file name to be stored (default: ./data.h5)
    """
    if not filename[-3:] == '.h5':
        filename += '.h5'

    # if os.path.isfile(filename):
        # print("! modifying existing file {} !".format(filename))

    if type(data_dict) is not dict:
        data_dict = _to_dict(data_dict)

    with hdf.File(filename, "a") as f:
        if group_name in f.keys():
            print("! overwriting group {} !".format(group_name))
            del f[group_name]

        group = f.create_group(group_name)
        _dict_to_group_recursive(group, data_dict)

    print(" > {} written to {} !".format(group_name, filename))
    # return filename


def _dict_to_group_recursive(group, data_dict):
    """ in the given group, store the data_dict provided.
    recurses through the data_dict and creates new groups and attributes
    """
    for att in data_dict.keys():
        dtt = data_dict[att]
        if type(dtt) == dict:
            subgroup = group.create_group(att)
            _dict_to_group_recursive(subgroup, dtt)
        else:
            if dtt is None:
                dtt = np.string_('None')
            if   (type(dtt) is str):
                dt = hdf.special_dtype(vlen=str)
                ds = group.create_dataset(att,(1,), dtype=dt)
                ds[0] = dtt
            elif (type(dtt) is tuple):
                group.attrs.create(att, dtt)
            elif (np.array(dtt).size < 12):
                group.attrs.create(att, dtt)
            else:
                group.create_dataset(att, data = dtt)


def _group_to_dict_recursive(elem, subelems = None):
    """ for a given element, return the value.  if group, traverse through the
    group and return dictionary with attributes and contained subelems"""
    if (type(elem) is hdf._hl.group.Group) or (type(elem) is hdf._hl.files.File):
        data = dict()
        if subelems == None: # no elem specified, read all
            subelems = elem.keys()
        elif type(subelems) is not list:
            subelems = [subelems]
        for sub in subelems:
            if elem.get(sub) is None: #scan for match in substrings
                ek = elem.keys()
                sub_matches = [match for match in ek if sub in match]
                # print(sub, " matches ", sub_matches)
                for match in sub_matches:
                    subdata = _group_to_dict_recursive(elem[match])
                    data.update({ match : subdata })
            else:
                subdata = _group_to_dict_recursive(elem[sub])
                data.update({ sub : subdata })
        data.update({aa:bb for aa,bb in elem.attrs.items()})
        # scan for Nones
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data.update({key:None for key,value in data.items() if value == b'None'})
        return data

    elif (type(elem) is hdf._hl.dataset.Dataset): # read_data
        if hasattr(np.array(elem)[0],'decode'):
            return np.array(elem)[0].decode('utf-8')
        else:
            return np.array(elem)
    else:
        print('hl type unknown, skip')


def read_from_hdf(filename = './autosave', groups = None):
    if not filename[-3:] == '.h5':
        filename += '.h5'

    if (groups is not None) and (type(groups) is not list):
        groups = [groups]

    try:
        with hdf.File(filename, 'r+') as f:
            data = _group_to_dict_recursive(f, groups)
    except:
        import gc
        for obj in gc.get_objects():   # Browse through ALL objects
            if isinstance(obj, hdf.File):   # Just HDF5 files
                try:
                    obj.close()
                    print("CLOSED")
                except:
                    pass # Was already closed

    if (groups is not None) and (len(groups) == 1):
        dk = data.keys()
        return data[list(dk)[0]]
    else:
        return data


