import numpy
import h5py
import logging
from pybosaris import Ndx, Key
from pybosaris.libmath import ismember
from pybosaris.sidekit_wrappers import check_path_existance


__author__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@eurecom.fr"
__credit__ = ["Niko Brummer", "Edward de Villiers", "Anthonay Larcher"]


class Quality:
    """
    Storage of quality information (measures, estimates, ...).
    Quality information is sample based (no-reference quality), not comparison based (relative quality).
    Aligning several quality objects with an ndx will result in comparability with another.
    Quality values summarize the quality of a sample. (A detailed level, like with SampleQuality, is not provided.)
    """
    def __init__(self, filename=''):
        self.modelQ = numpy.array([])
        self.segQ = numpy.array([])
        self.modelset = numpy.empty(0, dtype="|O")
        self.segset = numpy.empty(0, dtype="|O")
        self.hasmodel = numpy.array([], dtype="bool")
        self.hasseg = numpy.array([], dtype="bool")
        self.scoremask = numpy.array([], dtype="bool")

        if filename == '':
            pass
        else:
            tmp = Quality.read(filename)
            self.modelQ = tmp.modelQ
            self.segQ = tmp.segQ
            self.modelset = tmp.modelset
            self.segset = tmp.segset
            self.hasmodel = tmp.hasmodel
            self.hasseg = tmp.hasseg
            self.scoremask = tmp.scoremask

    def __repr__(self):
        ch = 'modelset:\n'
        ch += self.modelset + '\n'
        ch += 'segset:\n'
        ch += self.segset + '\n'
        ch += 'hasmodel:\n'
        ch += self.hasmodel.__repr__() + '\n'
        ch += 'hasseg:\n'
        ch += self.hasseg.__repr__() + '\n'
        ch += 'scoremask:\n'
        ch += self.scoremask.__repr__() + '\n'
        ch += 'modelQ:\n'
        ch += self.modelQ.__repr__() + '\n'
        ch += 'segQ:\n'
        ch += self.segQ.__repr__() + '\n'

    @check_path_existance
    def write(self, output_file_name):
        """ Save Scores in HDF5 format

        :param output_file_name: name of the file to write to
        """
        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("modelset", data=self.modelset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("segset", data=self.segset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("score_mask", data=self.scoremask.astype('int8'),
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("hasmodel", data=self.hasmodel.astype('int8'),
                             maxshape=(None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("hasseg", data=self.hasseg.astype('int8'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("modelQ", data=self.modelQ,
                             maxshape=(None,None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("segQ", data=self.segQ,
                             maxshape=(None,None),
                             compression="gzip",
                             fletcher32=True)

    """
    @check_path_existance
    def write_txt(self, output_file_name):
        "" "Save a Scores object in a text file

        :param output_file_name: name of the file to write to
        "" "
        deserializer = pandas.DataFrame(self.__dict__.values(), self.__dict__.keys()).T
        deserializer.to_csv(output_file_name, sep=' ', index=None, header=None)
    """

    @check_path_existance
    def write_matlab(self, output_file_name):
        """Save a Scores object in Bosaris compatible HDF5 format

        :param output_file_name: name of the file to write to
        """
        with h5py.File(output_file_name, "w") as f:
            f.create_dataset("/ID/row_ids", data=self.modelset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("/ID/column_ids", data=self.segset.astype('S'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("score_mask", data=self.scoremask.astype('int8'),
                             maxshape=(None, None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("/ID/has_row", data=self.hasmodel.astype('int8'),
                             maxshape=(None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("/ID/has_column", data=self.hasseg.astype('int8'),
                             maxshape=(None,),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("modelQ", data=self.modelQ.T,  # matlab conformance
                             maxshape=(None,None),
                             compression="gzip",
                             fletcher32=True)
            f.create_dataset("segQ", data=self.segQ.T,  # matlab conformance
                             maxshape=(None,None),
                             compression="gzip",
                             fletcher32=True)

    def align_with_ndx(self, ndx):
        """The ordering in the output Scores object corresponds to ndx, so
        aligning several Scores objects with the same ndx will result in
        them being comparable with each other.

        :param ndx: a Key or Ndx object

        :return: resized version of the current Scores object to size of \'ndx\'
                and reordered according to the ordering of modelset and segset in \'ndx\'.
        """
        assert(isinstance(ndx, Key) or isinstance(ndx, Ndx))
        assert(self.validate())
        assert(ndx.validate())

        aligned_obj = Quality()
        aligned_obj.modelset = ndx.modelset
        aligned_obj.segset = ndx.segset

        hasmodel = numpy.array(ismember(ndx.modelset, self.modelset))
        rindx = numpy.array([numpy.argwhere(self.modelset == v)[0][0] for v in ndx.modelset[hasmodel]]).astype(int)
        hasseg = numpy.array(ismember(ndx.segset, self.segset))
        cindx = numpy.array([numpy.argwhere(self.segset == v)[0][0] for v in ndx.segset[hasseg]]).astype(int)

        aligned_obj.hasmodel = hasmodel
        aligned_obj.hasseg = hasseg
        aligned_obj.hasmodel[hasmodel] = self.hasmodel[rindx]
        aligned_obj.hasseg[hasseg] = self.hasseg[cindx]

        assert(numpy.isfinite(self.modelQ).all())
        assert(numpy.isfinite(self.segQ).all())


        if self.modelQ.ndim == 1:
            aligned_obj.modelQ = numpy.zeros((ndx.modelset.shape[0]))
            aligned_obj.modelQ[numpy.where(hasmodel)[0]] = self.modelQ[rindx]
            aligned_obj.segQ = numpy.zeros((ndx.segset.shape[0]))
            aligned_obj.segQ[numpy.where(hasseg)[0]] = self.segQ[cindx]
        else:
            aligned_obj.modelQ = numpy.zeros((ndx.modelset.shape[0], self.modelQ.shape[1]))
            aligned_obj.modelQ[numpy.where(hasmodel)[0], :] = self.modelQ[rindx[numpy.where(hasmodel)[0]], :]
            aligned_obj.segQ = numpy.zeros((ndx.segset.shape[0], self.segQ.shape[1]))
            aligned_obj.segQ[numpy.where(hasseg)[0], :] = self.segQ[cindx[numpy.where(hasseg)[0]], :]

        aligned_obj.scoremask = numpy.zeros((ndx.modelset.shape[0], ndx.segset.shape[0]), dtype='bool')
        aligned_obj.scoremask[hasmodel[:, None] @ hasseg[None, :]] = self.scoremask[rindx[numpy.where(hasmodel)[0], None], cindx[None, numpy.where(hasseg)[0]]].flatten()

        assert numpy.sum(aligned_obj.scoremask) <= (numpy.sum(hasmodel) * numpy.sum(hasseg)), 'Error in new scoremask'

        if isinstance(ndx, Ndx):
            aligned_obj.scoremask = aligned_obj.scoremask & ndx.trialmask
        else:
            aligned_obj.scoremask = aligned_obj.scoremask & (ndx.tar | ndx.non)

        if numpy.sum(hasmodel) < ndx.modelset.shape[0]:
            logging.info('models reduced from %d to %d', ndx.modelset.shape[0], numpy.sum(hasmodel))
        if numpy.sum(hasseg) < ndx.segset.shape[0]:
            logging.info('testsegs reduced from %d to %d', ndx.segset.shape[0], numpy.sum(hasseg))

        if isinstance(ndx, Key):
            tar = ndx.tar & aligned_obj.scoremask
            non = ndx.non & aligned_obj.scoremask

            missing = numpy.sum(ndx.tar) - numpy.sum(tar)
            if missing > 0:
                logging.info('%d of %d targets missing', missing, numpy.sum(ndx.tar))
            missing = numpy.sum(ndx.non) - numpy.sum(non)
            if missing > 0:
                logging.info('%d of %d non targets missing', missing, numpy.sum(ndx.non))

        else:
            mask = ndx.trialmask & aligned_obj.scoremask
            missing = numpy.sum(ndx.trialmask) - numpy.sum(mask)
            if missing > 0:
                logging.info('%d of %d trials missing', missing, numpy.sum(ndx.trialmask))

        assert aligned_obj.validate(), 'Wrong Quality format'
        return aligned_obj

    def validate(self):
        """Checks that an object of type Scores obeys certain rules that
        must always be true.

            :return: a boolean value indicating whether the object is valid.
        """
        ok = self.modelQ.shape[1] == self.segQ.shape[1]
        ok &= (self.scoremask.shape[0] == self.modelset.shape[0] == self.hasmodel.shape[0] == self.modelQ.shape[0])
        ok &= (self.scoremask.shape[1] == self.segset.shape[0] == self.hasseg.shape[0] == self.segQ.shape[0])

        return ok

    def __eq__(self, other):
        assert(self.validate())
        assert(isinstance(other,Quality))
        assert(other.validate())

        ok = self.scoremask.shape == other.scoremask.shape
        ok &= self.modelQ.shape == other.modelQ.shape
        if not ok:
            return False

        sort_idx0_model = numpy.argsort(self.modelset)
        sort_idx0_segs = numpy.argsort(self.segset)
        sort_idx1_model = numpy.argsort(other.modelset)
        sort_idx1_segs = numpy.argsort(other.segset)

        ok = (self.modelset[sort_idx0_model] == other.modelset[sort_idx1_model]).all()
        ok &= (self.segset[sort_idx0_segs] == other.segset[sort_idx1_segs]).all()
        ok &= (self.hasmodel[sort_idx0_model] == other.hasmodel[sort_idx1_model]).all()
        ok &= (self.hasseg[sort_idx0_segs] == other.hasseg[sort_idx1_segs]).all()
        ok &= (self.scoremask[sort_idx0_model,:][:,sort_idx0_segs] == other.scoremask[sort_idx1_model,:][:,sort_idx1_segs]).all()

        if self.modelQ.ndim == 1:
            ok &= (self.modelQ[sort_idx0_model] == other.modelQ[sort_idx1_model]).all()
            ok &= (self.segQ[sort_idx0_segs] == other.segQ[sort_idx1_segs]).all()
        else:
            ok &= (self.modelQ[sort_idx0_model,:] == other.modelQ[sort_idx1_model,:]).all()
            ok &= (self.segQ[sort_idx0_segs,:] == other.segQ[sort_idx1_segs,:]).all()

        return ok

    @staticmethod
    def read(input_file_name):
        """Read a Scores object from information in a hdf5 file.

        :param input_file_name: name of the file to read from
        """
        with h5py.File(input_file_name, "r") as f:
            obj = Quality()

            obj.modelset = numpy.empty(f["modelset"].shape, dtype=f["modelset"].dtype)
            f["modelset"].read_direct(obj.modelset)
            obj.modelset = obj.modelset.astype('U100', copy=False)

            obj.segset = numpy.empty(f["segset"].shape, dtype=f["segset"].dtype)
            f["segset"].read_direct(obj.segset)
            obj.segset = obj.segset.astype('U100', copy=False)

            obj.scoremask = numpy.empty(f["score_mask"].shape, dtype=f["score_mask"].dtype)
            f["score_mask"].read_direct(obj.scoremask)
            obj.scoremask = obj.scoremask.astype('bool', copy=False)

            obj.hasmodel = numpy.empty(f["hasmodel"].shape, dtype=f["hasmodel"].dtype)
            f["hasmodel"].read_direct(obj.hasmodel)
            obj.hasmodel = obj.hasmodel.astype('bool', copy=False)

            obj.hasseg = numpy.empty(f["hasseg"].shape, dtype=f["hasseg"].dtype)
            f["hasseg"].read_direct(obj.hasseg)
            obj.hasseg = obj.hasseg.astype('bool', copy=False)

            obj.modelQ = numpy.empty(f["modelQ"].shape, dtype=f["modelQ"].dtype)
            f["modelQ"].read_direct(obj.modelQ)

            obj.segQ = numpy.empty(f["segQ"].shape, dtype=f["segQ"].dtype)
            f["segQ"].read_direct(obj.segQ)

            assert obj.validate(), "Error: wrong Quality format"
            return obj

    @staticmethod
    def read_matlab(input_file_name):
        """Read a Scores object from information in a hdf5 file in Matlab BOSARIS format.

            :param input_file_name: name of the file to read from
        """
        with h5py.File(input_file_name, "r") as f:
            obj = Quality()

            obj.modelset = numpy.empty(f["/ID/row_ids"].shape, dtype=f["/ID/row_ids"].dtype)
            f["/ID/row_ids"].read_direct(obj.modelset)
            obj.modelset = obj.modelset.astype('U100', copy=False)

            obj.segset = numpy.empty(f["/ID/column_ids"].shape, dtype=f["/ID/column_ids"].dtype)
            f["/ID/column_ids"].read_direct(obj.segset)
            obj.segset = obj.segset.astype('U100', copy=False)

            obj.scoremask = numpy.empty(f["score_mask"].shape, dtype=f["score_mask"].dtype)
            f["score_mask"].read_direct(obj.scoremask)
            obj.scoremask = obj.scoremask.astype('bool', copy=False)

            obj.hasmodel = numpy.empty(f["/ID/has_row"].shape, dtype=f["/ID/has_row"].dtype)
            f["/ID/has_row"].read_direct(obj.hasmodel)
            obj.hasmodel = obj.hasmodel.astype('bool', copy=False)

            obj.hasseg = numpy.empty(f["/ID/has_column"].shape, dtype=f["/ID/has_column"].dtype)
            f["/ID/has_column"].read_direct(obj.hasseg)
            obj.hasseg = obj.hasseg.astype('bool', copy=False)

            # obj.modelQ = numpy.empty(f["modelQ"].shape, dtype=f["modelQ"].dtype)
            # f["modelQ"].read_direct(obj.modelQ)
            obj.modelQ = f.get("modelQ").value.T

            # obj.segQ = numpy.empty(f["segQ"].shape, dtype=f["segQ"].dtype)
            # f["segQ"].read_direct(obj.segQ)
            obj.segQ = f.get("segQ").value.T

            assert obj.validate(), "Error: wrong Quality format"
            return obj

    """
    @staticmethod
    def read_txt(input_file_name):
        "" "Creates a Scores object from information stored in a text file.

        :param input_file_name: name of the file to read from
        "" "
        obj = Quality()
        serializer = pandas.read_csv(input_file_name, sep=' ', names=obj.__dict__.keys())
        for k in serializer.keys():
            dtype = obj.__getattribute__(k).dtype
            values = serializer[k].to_numpy(dtype=dtype)
            if dtype == object:
                values[serializer[k].isna()] = None
            obj.__setattr__(k, values)
        assert obj.validate(), "Wrong Quality format"
        return obj
    """

    """
    @staticmethod
    def merge(quality_list):
        "" "Merges a list of Quality objects. Assumption: they are disjoint.

        :param quality_list: the list of Scores object to merge
        " ""
        assert isinstance(quality_list, list), "Input is not a list"
        for obj in quality_list:
            assert isinstance(obj, Quality), \
                '{} {} {}'.format("Element ", obj, " is not a Quality")
            assert(obj.validate())

        merge_obj = Quality()
        for obj in quality_list:
            merge_obj.modelset = numpy.concatenate((merge_obj.modelset, obj.modelset))
            merge_obj.segset = numpy.concatenate((merge_obj.segset, obj.segset))

            merge_obj.modelQ = numpy.concatenate((merge_obj.modelQ, obj.modelQ))
            merge_obj.segQ = numpy.concatenate((merge_obj.segQ, obj.segQ))

            merge_obj.scoremask = numpy.array(
                [merge_obj.scoremask, numpy.zeros((merge_obj.modelset[0],obj.segset.shape[0]),dtype=bool),
                numpy.zeros((obj.modelset.shape[0],obj.segset.shape[0]),dtype=bool), obj.scoremask]
            )

            merge_obj.hasmodel = numpy.concatenate((merge_obj.hasmodel, obj.hasmodel))
            merge_obj.hasseg = numpy.concatenate((merge_obj.hasseg, obj.hasseg))

        assert merge_obj.validate(), 'Wrong Quality format'
        return merge_obj
    """
