import os
import sys
import json
import shutil
import numpy as np
from collections import OrderedDict


def main(json_dir, src_dir, tar_dir):
    with open(json_dir, 'r') as f:
            json_file = json.load(f)
    dataset = OrderedDict(json_file)

    total_num = 1000
    i= 1
    pitch_list = []
    instr_fml_str_list = []
    instr_id_list = []
    instr_str_list = []
    instrument_output = []


    for k, v in dataset.items():
        pitch_list.append(v['pitch'])
        instr_fml_str_list.append(v['instrument_family_str'])
        instr_id_list.append(v['instrument_family']*10+1)
        instr_str_list.append(v['instrument_str'])
        
        instrument_output.append(['nsynth', i, v['instrument_str']+'_'+str(v['pitch'])])
        src_file = os.path.join(src_dir, k +'.wav')
        tar_file = os.path.join(tar_dir, 'nsynth-' + str(i) + '.wav')
        assert os.path.isfile(src_file)
        shutil.copyfile(src_file, tar_file)

        i += 1
        if i > total_num:
            break

    print(instrument_output)
    print(pitch_list)
    print(instr_id_list)


if __name__== '__main__':
    json_dir = sys.argv[1]
    src_dir = sys.argv[2]
    tar_dir = sys.argv[3]
    main(json_dir, src_dir, tar_dir)