import numpy as np
import threading
from . import callbacks as clb
import torch


class BaseTransformer:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwds):
        pass


class BaseGenerator(clb.BaseCallback):

    MAX_NUMBER_OF_THREADS = 8

    @property
    def available_threads(self):
        return BaseGenerator.MAX_NUMBER_OF_THREADS if BaseGenerator.MAX_NUMBER_OF_THREADS < 3 \
                                                   else BaseGenerator.MAX_NUMBER_OF_THREADS-1

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        # for paralelizm
        self.thread = None
        self.thread_data = (-1, None) # (idx, data)

    def __getstate__(self):
        """
        Return dict without thread object and it's data
        """
        d = self.__dict__.copy()
        # save as reset
        d['thread'] = None
        d['thread_data'] = (-1, None)

        return d

    def __getitem__(self, idx, debug=False):
        # while debug mode run sequential version
        if debug or BaseGenerator.MAX_NUMBER_OF_THREADS < 3:
            data = self.load_data(idx, debug)
        else:
            # load data with paralellism
            data =  self.get_data(idx)

        return data

    # method to overload
    def __len__(self):
        raise Exception('This method should be overloaded')

    # method to overload
    def load_data(self, idx, debug=False):
        """
        Method should return data
        for idx-th batch of data.
        If idx is out of range method should return None.

        :param idx: index of batch to load
        :param debug: debug mode could give specific results

        :retrun data: data for idx-th batch or None if out of range
        """
        raise NotImplementedError('This method should be overloaded for sequential parallelism of data loading')

    def get_data(self, idx):
        """
        Method return data for idx by parallel.
        if idx-th data loading were started earlier, these data are returned,
        else data are loaded by this thread sequentially.

        Method start loading next batch in parallel.

        :param idx: index of batch to load

        :return data: loaded data for idx-th batch
        """
        # wait for thread if run
        self._join()
        # check idx of loaded data
        if self.thread_data[0] == idx:
            data = self.thread_data[1]
        # loaded previously data is not appropriate
        else:
            # load idx-th batch of data by this main thread
            data = self.load_data(idx)

        # start loading ext batch
        self.load_data_parallel(idx+1)

        return data

    def _join(self):
        """
        Method join existing thread with main thread.
        """
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def load_data_parallel(self, idx):
        """
        Method loads idx-th batch of data into self.thread_data,
        without check if data were loaded previously.
        Method does not return anything directly.
        To obtain data from parallel thread, thread must by joined by self._join()
        and then returned data are available by self.thread_data.

        :param idx: index of batch to load

        :return : None
        """
        # if previous were not joined / finished
        if self.thread is not None:
            self._join()
        # create new thread
        thread = threading.Thread(target=self._thread_function,
                                  args=(idx,))
        # run it without wait
        thread.start()
        # and save
        self.thread = thread

    def _thread_function(self, idx):
        """
        Method call method load_data in parallel thread
        :param idx: index of batch to load
        """
        data = self.load_data(idx)
        # save data with idx
        self.thread_data = (idx, data)

    def _parallel_loading_init(self):
        """
        Method initialize parallel loading
        """
        # run get first batch
        if BaseGenerator.MAX_NUMBER_OF_THREADS > 2:
            self.load_data_parallel(0)

    def _parallel_loading_end(self):
        """
        Method finish parallel loading
        """
        # join last thread
        self._join()
        # reset thread data
        self.thread_data = (-1, None)

    def reset(self):
        super().reset()
        # self._join()
        if self.thread is not None:
            self.thread.join(3)
            self.thread = None

        self.thread_data = (-1, None)

    def on_epoch_begin(self, *args, **kwargs):
        self.reset()
        self._parallel_loading_init()

    def on_training_begin(self, *args, **kwargs):
        self.reset()
        self._parallel_loading_init()

    def on_training_end(self, *args, **kwargs):
        self._parallel_loading_end()

    # used for validation and test set evaluation
    def on_score_begin(self, *args, **kwargs):
        self.reset()
        self._parallel_loading_init()

    def on_score_end(self, *args, **kwargs):
        self._parallel_loading_end()

    def on_epoch_end(self, *args, **kwargs):
        self._parallel_loading_end()
