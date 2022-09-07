import numpy
import h5py
import copy
import sys
import logging
import pandas
from pybosaris.libmath import ismember
from pybosaris.sidekit_wrappers import check_path_existance


__author__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@eurecom.fr"
__credit__ = ["Niko Brummer", "Edward de Villiers", "Anthonay Larcher"]


class SampleQuality:
    """
    A class to map sample IDs (of reference/probe samples) with the (measured, estimated, ...) quality of a sample.
    Using start/stop, frame-wise annotation (e.g., see speaker diarization) are facilitated.

    based on sidekit.bosaris.SampleQuality
    """
    def __init__(self, filename=''):
        self.sample_ids = numpy.empty(0, dtype="|O")
        self.quality = numpy.array([])
        self.start = numpy.empty(0, dtype="|O")
        self.stop = numpy.empty(0, dtype="|O")

        if filename == '':
            pass
        else:
            tmp = SampleQuality.read(filename)
            self.sample_ids = tmp.sample_ids
            self.quality = tmp.quality
            self.start = tmp.start
            self.stop = tmp.stop


    def __repr__(self):
        ch = '-' * 30 + '\n'
        ch += 'sample ids:' + self.sample_ids.__repr__() + '\n'
        ch += 'quality:' + self.quality.__repr__() + '\n'
        ch += 'seg start:' + self.start.__repr__() + '\n'
        ch += 'seg stop:' + self.stop.__repr__() + '\n'
        ch += '-' * 30 + '\n'
        return ch

    @check_path_existance
    def write(self, output_file_name):
        """ Save SampleQuality in HDF5 format

        :param output_file_name: name of the file to write to
        """
        assert self.validate(), "Error: wrong SampleQuality format"
        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("ids", data=self.sample_ids.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            if self.quality.ndim == 1:
                f.create_dataset("values", data=self.quality,
                                 maxshape=(None),
                                 compression="gzip",
                                 fletcher32=True)
            else:
                f.create_dataset("values", data=self.quality.T,  # for matlab consistency
                                 maxshape=(None, None),
                                 compression="gzip",
                                 fletcher32=True)
            # WRITE START and STOP
            start = copy.deepcopy(self.start)
            start[numpy.isnan(self.start.astype('float'))] = -1
            start = start.astype('int32', copy=False)

            stop = copy.deepcopy(self.stop)
            stop[numpy.isnan(self.stop.astype('float'))] = -1
            stop = stop.astype('int32', copy=False)

            f.create_dataset("start", data=start,
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("stop", data=stop,
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)

    @check_path_existance
    def write_txt(self, output_file_name):
        """Saves the SampleQuality to a text file.

        :param output_file_name: name of the output text file
        """
        deserializer = pandas.DataFrame(self.__dict__.values(), self.__dict__.keys()).T
        deserializer.to_csv(output_file_name, sep=' ', index=None, header=None)

    def validate(self, warn=False):
        """Checks that an object of type SampleQuality obeys certain rules that
        must alows be true.

        :param warn: boolean. If True, print a warning if strings are
            duplicated in either left or right array

        :return: a boolean value indicating whether the object is valid.

        """
        ok = (self.sample_ids.shape[0] == self.quality.shape[0] == self.start.shape[0] == self.stop.shape[0]) & self.sample_ids.ndim == 1

        if warn & (self.sample_ids.shape != numpy.unique(self.sample_ids).shape):
            logging.warning('The sample id list contains duplicate identifiers')
        if warn & (self.quality.shape != numpy.unique(self.quality).shape):
            logging.warning('The quality contains duplicate identifiers')
        return ok

    def __eq__(self, other):
        assert(self.validate())
        assert(isinstance(other,SampleQuality))
        assert(other.validate())

        ok = self.sample_ids.shape == other.sample_ids.shape
        if not ok:
            return False

        ok = (self.sample_ids == other.sample_ids).all()
        ok &= (numpy.abs(self.quality - other.quality) < 1e-15).all()
        ok &= (self.start == other.start).all()
        ok &= (self.stop == other.stop).all()

        return ok

    def set(self, sample_ids, quality, start=None, stop=None):
        self.sample_ids = copy.deepcopy(sample_ids)
        self.quality = copy.deepcopy(quality)

        if start is not None:
            self.start = copy.deepcopy(start)
        else:
            self.start = numpy.empty(self.sample_ids.shape, '|O')

        if stop is not None:
            self.stop = copy.deepcopy(stop)
        else:
            self.stop = numpy.empty(self.sample_ids.shape, '|O')

    @staticmethod
    def read(input_file_name):
        """Read SampleQuality in hdf5 format.

        :param input_file_name: name of the file to read from
        """
        with h5py.File(input_file_name, "r") as f:
            obj = SampleQuality()

            obj.sample_ids = f.get("ids").value
            obj.quality = f.get("values").value.T  # for matlab consistency

            # if running python 3, need a conversion to unicode
            if sys.version_info[0] == 3:
                obj.sample_ids = obj.sample_ids.astype('U255', copy=False)

            tmpstart = f.get("start").value
            tmpstop = f.get("stop").value
            obj.start = numpy.empty(f["start"].shape, '|O')
            obj.stop = numpy.empty(f["stop"].shape, '|O')
            obj.start[tmpstart != -1] = tmpstart[tmpstart != -1]
            obj.stop[tmpstop != -1] = tmpstop[tmpstop != -1]

            assert obj.validate(), "Error: wrong SampleQuality format"
            return obj

    @staticmethod
    def read_txt(input_file_name):
        """Read SampleQuality in text format.

        format examples (without start/stop)
        ------------------------------------
        for 1d quality: SAMPLE_ID VALUE ==> txfga:a 0.34224
        for 2+ d quality: SAMPLE_ID ["VALUES"] ==> txfga:a ["0.34224 0.994673 0.234173"]

        :param input_file_name: name of the file to read from
        """
        obj = SampleQuality()
        serializer = pandas.read_csv(input_file_name, sep=' ', names=obj.__dict__.keys())
        for k in serializer.keys():
            dtype = obj.__getattribute__(k).dtype
            values = serializer[k].to_numpy(dtype=dtype)
            if dtype == object:
                values[serializer[k].isna()] = None
            obj.__setattr__(k, values)


        if not obj.validate():
            raise Exception('Wrong format of SampleQuality')
        assert obj.validate(), "Error: wrong SampleQuality format"
        return obj

    @staticmethod
    def join(*obj_list, join='outer'):
        """ Joins multiple sample quality objects (e.g., with single dimensional SNR, duration, ...)
        assumption: same sample_ids.

        :param obj_list: A list of SampleQuality objects.
        :param join: join type, see pandas.concat documentation

        :return: an SampleQuality object that contains the joined information.
        """

        pd_idlets = []
        for obj in obj_list:
            assert(isinstance(obj, SampleQuality))
            assert(obj.validate())
            pd_obj = pandas.DataFrame.from_dict({'sample_ids': obj.sample_ids, 'quality': obj.quality, 'start': obj.start, 'stop': obj.stop}).set_index(['sample_ids', 'start', 'stop'])
            pd_idlets.append(pd_obj)

        pd_join = pandas.concat(pd_idlets, axis=1, join='outer')
        join_obj = SampleQuality()
        join_obj.sample_ids = pd_join.reset_index()['sample_ids'].values
        join_obj.quality = pd_join.to_numpy()
        join_obj.start = pd_join.reset_index()['start'].values
        join_obj.stop = pd_join.reset_index()['stop'].values

        if not join_obj.validate():
            raise Exception('Wrong format of SampleQuality')

        return join_obj

    def merge(self, sq2):
        """ Merges the current SampleQuality with another SampleQuality or a list of SampleQuality objects..

        :param sq2: Another SampleQuality object.

        :return: an SampleQuality object that contains the information from the two
            input SampleQuality.
        """
        assert(isinstance(sq2, SampleQuality))
        obj = SampleQuality()
        if self.validate() & sq2.validate():
            # create tuples of (model,seg) for both SampleQualitys for quick comparaison
            tup1 = [(mod, seg) for mod, seg in zip(self.sample_ids, self.quality)]
            tup2 = [(mod, seg) for mod, seg in zip(sq2.sample_ids, sq2.quality)]

            # Get indices of common sessions
            existing_sessions = set(tup1).intersection(set(tup2))
            # Get indices of sessions which are not common in idmap2
            idx_new = numpy.sort(numpy.array([idx for idx, sess in enumerate(tup2) if sess not in tup1]))
            if len(idx_new) == 0:
                idx_new = numpy.zeros(sq2.sample_ids.shape[0], dtype='bool')

            obj.sample_ids = numpy.concatenate((self.sample_ids, sq2.sample_ids[idx_new]), axis=0)
            obj.quality = numpy.concatenate((self.quality, sq2.quality[idx_new]), axis=0)
            obj.start = numpy.concatenate((self.start, sq2.start[idx_new]), axis=0)
            obj.stop = numpy.concatenate((self.stop, sq2.stop[idx_new]), axis=0)

        else:
            raise Exception('Cannot merge SampleQuality, wrong type')

        if not obj.validate():
            raise Exception('Wrong format of SampleQuality')

        return obj

    def align_with_ids(self, ids):
        """
        Ordering of the ids in the output object corresponds to ids,
        so aligning several SampleQuality objects makes them easier comparable with another.

        :param ids: array of strings
        :return: SampleQuality with aligned IDs
        """
        assert(isinstance(ids, numpy.ndarray))
        assert(self.validate())
        assert(numpy.isfinite(self.quality).all())

        aligned_qual = SampleQuality()
        aligned_qual.sample_ids = copy.deepcopy(ids)

        hasids = numpy.array(ismember(ids, self.sample_ids))
        rindx = numpy.array([numpy.argwhere(self.sample_ids == v)[0][0] for v in ids[hasids]]).astype(int)

        if self.quality.ndim == 1:
            aligned_qual.quality = numpy.empty((ids.shape[0]))
            aligned_qual.quality[numpy.where(hasids)[0]] = self.quality[rindx]
        else:
            aligned_qual.quality = numpy.empty((hasids.sum(), self.quality.shape[1]))
            aligned_qual.quality[numpy.where(hasids)[0][:, None], :] = self.quality[rindx[:, None], :]
        aligned_qual.start = numpy.empty((ids.shape[0]))
        aligned_qual.start[numpy.where(hasids)[0]] = self.start[rindx]
        aligned_qual.stop = numpy.empty((ids.shape[0]))
        aligned_qual.stop[numpy.where(hasids)[0]] = self.stop[rindx]

        lost = self.quality.shape[0] - aligned_qual.quality.shape[0]
        if lost > 0:
            logging.info('Number of segments reduced from %d to %d', self.quality.shape[0], aligned_qual.quality.shape[0])

        usedids = hasids.sum()
        if usedids < ids.shape[0]:
            logging.info("%d of %d ids don't have quality values", ids.shape[0] - usedids, ids.shape[0])\

        return aligned_qual
