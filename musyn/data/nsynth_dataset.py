import json
import os
import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torchaudio
from torchaudio.compliance.kaldi import mfcc
from musyn.data.base_dataset import BaseDataset
from musyn.data.utils import lable_to_enc_dict

INSTR = [*range(1006)]
INSTR_FML = [*range(11)]

random.seed(1024)
np.random.seed(1024)
torchaudio.set_audio_backend('sox_io')


class NSynthDataset(BaseDataset):
    """A dataset class for NSynthDataset.
    """
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.datalist = list()
        self.dataset = dict()
        self.read_file()
        self.update_dataset()


    def read_file(self):
        """Read meta data of the dataset
        """
        # read examples.json
        with open(os.path.join(self.root, self.opt.datafile), 'r') as f:
            json_file = json.load(f)
        self.dataset = OrderedDict(json_file)
        self.datalist = list(self.dataset.keys())


    def update_dataset(self):
        """Transform the meta data to the format we want
        """
        # Instrument Encoder
        Instr_DICT = lable_to_enc_dict(INSTR, False)
        Instr_DICT_ONEHOT = lable_to_enc_dict(INSTR, True)

        # Instrument Family Encoder
        InstrFml_DICT = lable_to_enc_dict(INSTR_FML, False)
        InstrFml_DICT_ONEHOT = lable_to_enc_dict(INSTR_FML, True)

        self.instr_audio_dict = defaultdict(list)

        for k, v in self.dataset.items():
            # self.instr_list.append(v['instrument'])
            if self.opt.encode_cat:
                v['instr_enc'] = Instr_DICT[v['instrument']]
                v['instr_fml_enc'] = InstrFml_DICT[v['instrument_family']]
            if self.opt.one_hot_instr or self.opt.one_hot_all:
                v['instr_enc_onehot'] = Instr_DICT_ONEHOT[v['instrument']]
            if self.opt.one_hot_instr_family or self.opt.one_hot_all:
                v['instr_fml_enc_onehot'] = InstrFml_DICT_ONEHOT[v['instrument_family']]
            self.dataset[k].update(v)
            # add audio samples for instrument
            self.instr_audio_dict[v['instrument']].append(k)


    def stitch(self, note_str, note_info):
        max_time = self.opt.augment.max_sec * self.opt.sr[0]
        samples_pool = self.instr_audio_dict[note_info['instrument']]
        audio_len = 0
        data_file = os.path.join(self.root, 'trim_audio', note_str+'.wav')
        wav, _ = torchaudio.load(data_file)
        wavform = [wav]
        audio_len += wav.shape[1]
        while audio_len < max_time:
            note_str = random.sample(samples_pool, 1)[0]
            data_file = os.path.join(self.root, 'trim_audio', note_str+'.wav')
            wav, _ = torchaudio.load(data_file)
            wavform.append(wav)
            audio_len += wav.shape[1]
        wavform = torch.cat(wavform, dim=1)
        return wavform


    def __len__(self):
        return len(self.datalist)


    def __getitem__(self, index):
        note_str = self.datalist[index]
        note_info = self.dataset[note_str]

        # augment or not
        if self.opt.is_augment and self.opt.augment.type == 'stitch' and index%2==0:
            wavform = self.stitch(note_str, note_info)
        else:
            data_file = os.path.join(self.root, 'audio', note_str+'.wav')
            wavform, _ = torchaudio.load(data_file)

        # transform feat or not
        if self.opt.feat_type == 'mfcc':
            feat = mfcc(wavform, frame_length=25, low_freq=10, high_freq=8000, 
                num_mel_bins=80, num_ceps=80, snip_edges=False)
        else:
            feat = wavform

        if self.opt.is_train:
            instr = note_info['instr_enc']
            instr_fml = note_info['instr_fml_enc']
        else:
            instr = note_info['instrument']
            instr_fml = note_info['instrument_family']

        return {'feat': feat, 
                'label_A': instr, 'label_B': instr_fml, }

