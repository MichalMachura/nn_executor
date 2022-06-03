from typing import List, Tuple
from . import utils
import numpy as np
import numpy.random as rnd
import cv2 as cv
import os
from threading import Thread
import threading
from queue import Queue
from . import callbacks as clb
import torch


class BaseTransformer:
    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwds):
        pass


class YOLOTransformer(BaseTransformer):

    def __init__(self, generator=rnd.default_rng(),**kwargs):
        """
        Initialization of Transformer object
        :param self: -
        :param generator: np.random.Generator object, 
                          default np.random.default_rng()
        :param kwargs: key-word type arguments where: key - name of method/transformation,
                        word-parameters of key-method
        """
        super().__init__()
        self.rand_generator = generator
        # names of transformation as np.array
        self.operations = np.array(list(kwargs))
        self.transform_dict = {}

        # format transformation args into tuples
        for k,v in kwargs.items():
            # if not tuple
            if type(v) is not tuple:
                self.transform_dict[k] = (v,)
            else:
                self.transform_dict[k] = v

    def __call__(self, X, Y):
        """
        Apply transformation on image and it's bbox.
        Number of transformation and choice are random.
        :param self: -
        :param : X - image NxMx3, floating point [0.0;1.0] !!!
        :param : y - bbox np.array of size (4,)
        :return : X, y after applied transfomations
        """
        # get number of operation to execute on img
        num = self.rand_generator.integers(0,len(self.transform_dict.keys())+1)
        # num = len(self.transform_dict.keys())
        # get operations names to exec
        keys = self.rand_generator.choice(self.operations,size=num, replace=False)
        Y_OUT = Y.copy()
        # execute operations with update of image and bbox
        for k in keys:
            X, Y_OUT = getattr(self, k)(X, Y_OUT,*self.transform_dict[k])
        
        return X, Y_OUT

    def rotate(self, X, Y, angle_min, angle_max, *args):
        """
        Image rotation around FIRST object center by random angle 
        from range [angle_min; angle_max].
        :param self: -
        :param X : image np.array of shape NxM x3
        :param Y: bbox np.array of shape (n,4)
        :param angle_min: begin of angle range [deg]
        :param angle_max: end of angle range [deg]
        :param *args: anything
        :return : X, y after rotation
        """
        angle_deg = (self.rand_generator.random()*(angle_max-angle_min) + angle_min)
        angle = -np.deg2rad(angle_deg)
        
        ORIGINAL_ROW,ORIGINAL_COL = X.shape[:2]
        ORIGINAL_ROT = np.array([ORIGINAL_COL//2,ORIGINAL_ROW//2])
        ORIGINAL_SIZE = np.array([ORIGINAL_COL,ORIGINAL_ROW])
        ORIGINAL_D = int(np.sqrt(ORIGINAL_COL**2+ORIGINAL_ROW**2))
        TMP = np.zeros((ORIGINAL_D,ORIGINAL_D,X.shape[-1],),
                       dtype=X.dtype)
        EXTENDED_ROT = np.array([ORIGINAL_D/2,ORIGINAL_D/2])
        X_OFFSET = (ORIGINAL_D-X.shape[-2])//2
        Y_OFFSET = (ORIGINAL_D-X.shape[-3])//2
        OFFSET = np.array([X_OFFSET,Y_OFFSET])
        TMP[Y_OFFSET:Y_OFFSET+X.shape[-3],
            X_OFFSET:X_OFFSET+X.shape[-2], :] = X
        rot_mat = cv.getRotationMatrix2D(tuple(EXTENDED_ROT.tolist()), 
                                         angle_deg, 
                                         1.0)
        X = cv.warpAffine(TMP, rot_mat, (ORIGINAL_D, ORIGINAL_D))
        # objects corners description
        W_2 = Y[:,2]/2
        H_2 = Y[:,3]/2
        CORNER_ANGLE = np.arctan2(H_2, W_2)
        CORNER_R = np.sqrt(W_2*W_2 + H_2*H_2)
        # vec of obj center from point of rotation
        D_POS = Y[:,:2] + OFFSET - EXTENDED_ROT
        DX = D_POS[:,0]
        DY = D_POS[:,1]
        POS_ANGLE = np.arctan2(DY,DX)
        R_POS = np.sqrt(DX*DX + DY*DY)
        # change pos by rotation angle in new origin
        NEW_POS_ANGLE = POS_ANGLE+angle
        NEW_X = EXTENDED_ROT[0] + R_POS*np.cos(NEW_POS_ANGLE)
        NEW_Y = EXTENDED_ROT[1] + R_POS*np.sin(NEW_POS_ANGLE)
        # box corner rotation
        CORNER_ANGLES = np.concatenate([CORNER_ANGLE.reshape((-1,1))+angle,
                                        -CORNER_ANGLE.reshape((-1,1))+angle], axis=1)
        CORNER_X = np.cos(CORNER_ANGLES)
        CORNER_Y = np.sin(CORNER_ANGLES)
        # new box WH
        W = 2*CORNER_R*np.max(np.abs(CORNER_X),axis=1)
        H = 2*CORNER_R*np.max(np.abs(CORNER_Y),axis=1)
        
        Y_OUT = np.concatenate([NEW_X.reshape((-1,1)),
                                NEW_Y.reshape((-1,1)),
                                W.reshape((-1,1)),
                                H.reshape((-1,1))],axis=1)
        
        # GET bbox for all boxes
        Y_LTRB = utils.xcycwh_to_ltrb(Y_OUT.copy())
        # border of bboxes
        MIN_POS = np.min(Y_LTRB[:,:2], axis=0).astype(int)
        MAX_POS = np.max(Y_LTRB[:,2:], axis=0).astype(int)
        SIZE = MAX_POS - MIN_POS
        MASS_CENTER = (MIN_POS + MAX_POS) // 2
        SIZE = np.where(SIZE > ORIGINAL_SIZE, SIZE, ORIGINAL_SIZE)
        # determine beg-end
        beg = MASS_CENTER - SIZE // 2
        beg = np.where(beg < 0, 0, beg)
        end = beg + SIZE
        end = np.where(end > ORIGINAL_D, ORIGINAL_D, end)
        beg = end - SIZE
        # crop
        X = X[beg[1]:end[1],beg[0]:end[0],:]
        # sub offset
        Y_OUT[:,:2] -= beg
        
        # rescale if needed
        # if at least one dim is higher than original image
        if np.sum(SIZE > ORIGINAL_SIZE) > 0:
            SCALE = ORIGINAL_SIZE / SIZE
            X = cv.resize(X,tuple(ORIGINAL_SIZE.tolist()))
            Y_OUT[:,:2] *= SCALE
            Y_OUT[:,2:] *= SCALE
        
        
        return X, Y_OUT

    def scale(self, X, Y, scale_min, scale_max, *args):
        """
        Image scale by random scale factor 
        from range [scale_min; scale_max].
        If after scaling object defined by y, object size is larger than input 
        image size, applied is scale which allow for closing object size 
        inside image size.
        :param self: -
        :param X : image np.array of shape NxM x3
        :param Y: bbox np.array of shape (n,4)
        :param scale_min: begin of scale range > 0.0
        :param scale_max: end of scale range > 0.0
        :param *args: anything
        :return : X, y after scaling
        """
        # random interpolation
        interpolation = self.rand_generator.choice([cv.INTER_AREA, 
                                                    cv.INTER_NEAREST, 
                                                    cv.INTER_LANCZOS4, 
                                                    cv.INTER_LINEAR, 
                                                    cv.INTER_CUBIC])
        # random scale
        scales = self.rand_generator.random((2,))*(scale_max-scale_min) + scale_min
        # output array's shape
        ORIGINAL_SIZE = np.array(X.shape[:2][::-1])
        Y_LTRB = utils.xcycwh_to_ltrb(Y.copy())
        # border of bboxes
        MIN_POS = np.min(Y_LTRB[:,:2], axis=0).astype(int)
        MAX_POS = np.max(Y_LTRB[:,2:], axis=0).astype(int)
        MIN_SIZE = MAX_POS - MIN_POS
        MAX_SCALE = ORIGINAL_SIZE / MIN_SIZE
        # scale saturation
        scales = np.minimum(MAX_SCALE,scales)
        # dst shape
        NEW_SIZE = (ORIGINAL_SIZE*scales).astype(int)
        
        # resize img and rescale bboxes
        new_img = cv.resize(X, tuple(NEW_SIZE.tolist()), interpolation=interpolation)
        scales_4 = np.concatenate([scales.reshape((1,2)),
                                   scales.reshape((1,2))],
                                  axis=1).reshape((1,4))
        Y_OUT = Y*scales_4
        
        # zeroed for scales < 0
        X = np.zeros_like(X)
        # get center of all objects
        OBJs_CENTER = (((MIN_POS + MAX_POS)/2)*scales).astype(int) # XY
        # make right bottom borders of crop with center of all objects
        end = OBJs_CENTER + ORIGINAL_SIZE//2
        # limit by new img size
        end = np.where(end < NEW_SIZE, end, NEW_SIZE)
        # make left top borders of crop with center of all objects
        beg_expected = end - ORIGINAL_SIZE
        # limit by (0,0) point
        beg = np.where(beg_expected < 0, 0, beg_expected)
        
        # of beg was saturated end can be larger
        end = beg + ORIGINAL_SIZE
        # but if scale is < 1  => end is out of new img range
        # -> needed saturation once again
        end = np.where(end < NEW_SIZE, end, NEW_SIZE)
        # size of copied part
        EFFECTIVE_SIZE = end-beg
        # crop img with possible filling by zeros
        X[0:EFFECTIVE_SIZE[1],0:EFFECTIVE_SIZE[0],:] = new_img[beg[1]:end[1],beg[0]:end[0],:]
        # if beg was saturated -> obj offset is different
        Y_OUT[:,:2] -= beg

        return X, Y_OUT
    
    def blur(self, X, Y, max_kernel_size=1, *args):
        """
        Image blur with random kernel size from range [1;max_kernel_size]. 
        :param self: -
        :param X : image np.array of shape NxM x3
        :param Y: bbox np.array of size (n,4,)
        :param max_kernel_size: max size of kernel
        :param *args: anything
        :return : X, Y after blur
        """
        # get random filter size
        k_size = self.rand_generator.integers(1, max_kernel_size+1)
        k_size = (k_size//2)*2+1

        if k_size > 2:
            X = cv.blur(X, (k_size, k_size))

        return X, Y

    def equalize_hist(self, X, Y, probability=0.1, *args):
        """
        Image histograms equalizations. Transformation is executed with given 
        probaility.  
        :param self: -
        :param X : image np.array of shape NxM x3
        :param Y: bbox np.array of size (n,4,)
        :param probaility: probaility value of transformation execution
        :param *args: anything
        :return : X, Y after histograms equalizations
        """
        p = self.rand_generator.random()
        if p < probability:
            X = (X*255).astype(np.uint8)
            X[:,:,0] = cv.equalizeHist(X[:,:,0])
            X[:,:,1] = cv.equalizeHist(X[:,:,1])
            X[:,:,2] = cv.equalizeHist(X[:,:,2])
            X = X/255.0

        return X, Y
    
    def fill(self, X, Y, min_fill, max_fill, *args):    
        for bbox in Y:
            beg = (bbox[:2] - bbox[2:]/2).astype(int)
            beg = np.maximum(beg, 0)
            end = (bbox[:2] + bbox[2:]/2).astype(int)
            end = np.minimum(end, np.array(X.shape[:2][::-1]))
            WH = end-beg
            
            fill = min_fill + (max_fill-min_fill)*self.rand_generator.random((2,))
            fill_WH = (fill*WH).astype(int)
            fill_beg = self.rand_generator.integers(beg,end-fill_WH)
            fill_end = fill_beg+fill_WH
            X[fill_beg[1]:fill_end[1],fill_beg[0]:fill_end[0],:] = 0.5
            
        return X, Y
    
    def translate(self, X, Y, x_min_max, y_min_max,*args):    
        # random translation
        x_move = self.rand_generator.integers(*x_min_max)
        y_move = self.rand_generator.integers(*y_min_max)
        
        # bbox for all objects
        ORIGINAL_SIZE = np.array(X.shape[:2][::-1])
        Y_LTRB = utils.xcycwh_to_ltrb(Y.copy())
        # border of bboxes
        MIN_POS = np.min(Y_LTRB[:,:2], axis=0).astype(int)
        MAX_POS = np.max(Y_LTRB[:,2:], axis=0).astype(int)
        MIN_SIZE = MAX_POS - MIN_POS
        
        movement = np.array([x_move,y_move])
        movement = np.clip(movement,-MIN_POS,ORIGINAL_SIZE-MAX_POS)
        
        M = np.float32([[1, 0, movement[0]],
                        [0, 1, movement[1]]])
        X = cv.warpAffine(X, M, X.shape[:2][::-1])
        Y[:,:2] += movement.reshape((-1,2))

        return X, Y

    def HSV(self, X, Y, sigma,*args):
        X = cv.cvtColor(X, cv.COLOR_BGR2HSV)
        rand_noise = self.rand_generator.normal(loc=0, scale=sigma, size=X.shape[:2])
        # add noise to original image
        X[:,:,1] += rand_noise
        # avoid overflow
        np.clip(X, 0.0, 1.0)
        
        X = cv.cvtColor(X, cv.COLOR_HSV2BGR)
        return X, Y

    def LAB(self, X, Y, sigma,*args):
        X = cv.cvtColor(X, cv.COLOR_BGR2LAB)
        rand_noise = self.rand_generator.normal(loc=0, scale=sigma, size=X.shape)
        # add noise to original image
        X += rand_noise
        # avoid overflow
        np.clip(X, 0.0, 1.0)
        X = cv.cvtColor(X, cv.COLOR_LAB2BGR)
        
        return X, Y

    def YCrCb(self, X, Y, sigma,*args):
        X = cv.cvtColor(X, cv.COLOR_BGR2YCrCb)
        rand_noise = self.rand_generator.normal(loc=0, scale=sigma, size=X.shape[:2])
        # add noise to original image
        X[:,:,0] += rand_noise
        # avoid overflow
        np.clip(X, 0.0, 1.0)
        X = cv.cvtColor(X, cv.COLOR_YCrCb2BGR)
        
        return X, Y

    def horizontal_flip(self, X, Y, *args):
        """
        Image horizontal flip
        :param self: -
        :param X : image np.array of shape NxM x3
        :param y: bbox np.array of size (n,4)
        :param *args: anything
        :return : X, y after flip
        """
        X = np.fliplr(X)
        Y_OUT = []
        for y in Y:
            yy = y.copy()
            yy[0] = X.shape[1] - y[0]
            Y_OUT.append(yy)
        
        Y_OUT = np.array(Y_OUT)
        
        return X, Y_OUT

    def vertical_flip(self, X, Y, *args):
        """
        Image vertical flip
        :param self: -
        :param X : image np.array of shape NxM x3
        :param y: bbox np.array of size (n,4,)
        :param *args: anything
        :return : X, y after flip
        """
        X = np.flipud(X)
        
        Y_OUT = []
        for y in Y:
            yy = y.copy()
            yy[1] = X.shape[0] - y[1]
            Y_OUT.append(yy)
        
        Y_OUT = np.array(Y_OUT)
        
        return X, Y_OUT
    
    def dilate(self, X, Y, max_kernel_size, *args):
        """
        Apply on image maximal filter of random kernel size from 
        range [1; max_kernel_size]
        :param self: -
        :param X : image np.array of shape NxM x3
        :param y: bbox np.array of size (n,4,)
        :param *args: anything
        :param max_kernel_size: int, max size of dilate kernel 
        :return : X, Y after dilate
        """
        # get random filter size
        k_size = self.rand_generator.integers(1, max_kernel_size+1)
        k_size = (k_size//2)*2+1
        # create structural element
        se = np.ones((k_size, k_size), dtype=X.dtype)
        if k_size > 2:
            # exec dilation
            X = cv.dilate(X, se)
        
        return X, Y
    
    def erode(self, X, Y, max_kernel_size, *args):
        """
        Apply on image minimal filter of random kernel size from 
        range [1; max_kernel_size]
        :param self: -
        :param X : image np.array of shape NxM x3
        :param y: bbox np.array of size (n,4,)
        :param *args: anything
        :param max_kernel_size: int, max size of dilate kernel 
        :return : X, Y after erode
        """
        # get random filter size
        k_size = self.rand_generator.integers(1, max_kernel_size+1)
        k_size = (k_size//2)*2+1
        # create structural element
        se = np.ones((k_size, k_size), dtype=X.dtype)
        
        if k_size > 2:
            # exec erode
            X = cv.erode(X, se)
        
        return X, Y

    def noise(self, X, Y, sigma, *args):
        """
        Add to image noise with normal distribution of mean at 0 
        and standard deviation sigma.
        Expect X of type float normalized to [0.0;1.0].
        Output is saturated to range [0.0;1.0]
        :param self: -
        :param X : image np.array of shape NxM x3
        :param y: bbox np.array of size (n,4,)
        :param *args: anything
        :param sigma: standard deviation of noise 
        :return : X, Y after adding noise
        """
        rand_noise = self.rand_generator.normal(loc=0, scale=sigma, size=X.shape)
        # add noise to original image
        X += rand_noise
        # avoid overflow
        np.clip(X, 0.0, 1.0)

        return X, Y


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
        raise NotImplementedError('This method should be overloaded for sequetial parallelism of data loading')

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



def numpy_to_torch(device):

    def tmp(X, BBOXES, CLS, device=device):
        X = torch.tensor(np.moveaxis(X, 3, 1 ), 
                         dtype=torch.float32).contiguous().to(device)
        
        BBOXES_out = []
        CLS_out = []
        for bbox, cls in zip(BBOXES, CLS):
            BBOXES_out.append(torch.tensor(bbox, device=device) if bbox is not None else None)
            CLS_out.append(torch.tensor(cls, device=device) if cls is not None else None)
        
        return X, BBOXES_out, CLS_out

    return tmp
    

class YOLODataGenerator(BaseGenerator):
    
    def __init__(self,
                path_to_dataset:str,
                input_size,
                images_labels:List[Tuple[str,np.ndarray,np.ndarray]],
                batch_size:int,
                name:str='YoloDataGenerator', 
                augmentator:YOLOTransformer=None,
                rand_generator=rnd.default_rng(),
                after_load=numpy_to_torch,
                with_classes:bool=False,
                naive_resize=True
                ):
        """
        Initialize Generator object.
        :param self: -
        :param path_to_dataset: path to directory with dataset
        :param input_size:  size of network input 
        :param images_labels: list of tuples ('path_to_img', bboxes, classes)
                                both paths are relative to path_to_dataset
        :param batch_size: size of batch, int
        :param name: name of generator
        :param augmentator: object which allow for augmentation
                          - overloaded __call__(X, bboxes)
        :param rand_generator: np.random.Generator object, 
                               default np.ranodom.default_rng()
        :param after_load: function like X,BBOXES,CLS = after_load(X,BBOXES,CLS)
                            addition processing of X,BBOXES,CLS batch like numpy_to_torch
        """
        super().__init__(batch_size)
        
        self.images_labels = images_labels
        self.path_to_dataset = path_to_dataset
        self.input_size = input_size
        self.name = name
        self.augmentator = augmentator
        self.rand_generator = rand_generator
        self.after_load = after_load
        self.with_classes = with_classes
        self.naive_resize = naive_resize
        
    def load_data(self, idx, debug=False):
        """
        Get one batch of images with references outputs of each anchor's maps.
        Possible is return of batch bounsding box - for debug=True
        :param self: - 
        :param idx: int, index of batch
        :param debug: bool, flag default False
        :return: X - batch of images - np.array,
                 y - tuple of references for each img - 
                     (List[np.ndarray], List[np.ndarray]) - BBOXES, CLSs
        """
        if idx >= self.__len__():
            return None
        
        X, BBOXES, CLS = self.load_batch(idx)
        
        # possible additional operation on input and labels data like: convert numpy to tensor
        if self.after_load:
            X, BBOXES, CLS = self.after_load(X, BBOXES, CLS)
        
        return X, (BBOXES, CLS)
    
    def load_batch(self, idx):
        """
        Load batch of images with corresponding bboxes.
        If augmentator object is available, augmentation is applied 
        for each image(it's with bbox).
        Augmentation is executed by second thread, first thred load 
        data from disc to queue.
        :param self: - 
        :param idx: int, index of batch
        :return: X - np.array of shape batch_size x image_shape, 
                 y - np.array of shape batch_size x 4 [xc, yc, w, h]
                 CLS - np.array of shape batch_size x 1 
        """
        # indeces of begin and end of idx-batch
        batch_beg = idx*self.batch_size
        batch_end = min(batch_beg+self.batch_size, len(self.images_labels))
        # batch size have not to be the same at the end of data
        batch_size = max(0, batch_end - batch_beg)
        # init empty arrays
        X = np.empty((batch_size,self.input_size[1],self.input_size[0],3), dtype=np.float32)
        BBOXES:List[np.ndarray] = [None]*batch_size
        CLS:List[np.ndarray] = [None]*batch_size

        # create queue and fill with indeces
        queue = Queue(batch_size,)
        for i in range(batch_size):
            queue.put(i)

        def processing_fcn(self=self, 
                           queue=queue, 
                           batch_beg=batch_beg, 
                           X=X, 
                           BBOXES=BBOXES, 
                           CLS=CLS):
            while not queue.empty():
                try:
                    # get in-batch position
                    pos = queue.get()
                    # unlock queue for another thread
                    queue.task_done()
                except:
                    # get could raise an exception when queue is empty
                    # so going to next iteration allow for processing data, 
                    # when they appear in meantime
                    # if not main loop condition will break the loop
                    continue
                
                # index at images_labels
                index = batch_beg + pos
                # load image
                th_img = cv.imread(os.path.join(self.path_to_dataset, 
                                                self.images_labels[index][0]) )
                # copy all bboxes
                th_bbox = self.images_labels[index][1].copy()
                # and their classes if present, else set zero class for each bblox
                th_cls = self.images_labels[index][2].copy() if len(self.images_labels[index]) > 2 \
                                                             else np.zeros_like(th_bbox[:,0],dtype=np.longlong)
                
                # optional error printing
                if th_img is None or th_bbox is None:
                    raise RuntimeError("Error while reading {}-th element as img_path = {} ({}) record = {}".format(
                                        index,
                                        os.path.join(self.path_to_dataset, 
                                                     self.images_labels[index][0]), 
                                        *self.images_labels[index]))
                else:
                    # augmentation before resize
                    if self.augmentator:
                        # normalize
                        th_img = th_img.astype(np.float32) / 255
                        th_img, th_bbox = self.augmentator(th_img, th_bbox)
                    
                    H_in, W_in = th_img.shape[:2]
                    W_dst, H_dst = self.input_size
                    # if resize is necessary
                    if H_in != H_dst or W_in != W_dst:
                        # resize image 
                        if self.naive_resize:
                            th_img = cv.resize(th_img, 
                                               (W_dst, H_dst), 
                                               interpolation=cv.INTER_LINEAR)
                            W_new = W_dst
                            H_new = H_dst
                        else:
                            th_img, W_new, H_new = resize(th_img, 
                                                            W_dst, 
                                                            H_dst, 
                                                            INTER=cv.INTER_LINEAR)
                        # rescale bbox
                        scale = np.array([[W_new / W_in, H_new / H_in,W_new / W_in, H_new / H_in]])
                        th_bbox *= scale

                    # normalization if previously where not applied
                    if not self.augmentator:
                        th_img = th_img.astype(np.float32) / 255
                    
                    # save in dst out array / lists
                    X[pos,:,:,:] = th_img.astype(np.float32)
                    BBOXES[pos] = th_bbox.astype(np.float32)
                    CLS[pos] = th_cls

        # create threads and start them
        threads = [Thread(target=processing_fcn) for i in range(self.available_threads-1)]
        for th in threads:
            th.start()
        # processing also at main thread
        processing_fcn()
        # when main thread completed = > nothing to process
        # join all threads
        for th in threads:
            th.join()

        return X, BBOXES, CLS
        
    def __len__(self):
        """
        Return number of batches available for that generator
        :param self: -
        :return : number of batches as integer
        """
        return ((len(self.images_labels)-1)//self.batch_size) + 1
    
    def on_epoch_end(self, *args, **kwargs):
            """
            Method called to apply random images sequences
            :param self: -
            """
            super().on_epoch_end()
            self.rand_generator.shuffle(self.images_labels)


def resize(img:np.ndarray, W_DST:int, H_DST:int, INTER):
    H,W = img.shape[:2]

    # sizes after scale for each axis
    H1 = H * W_DST / W # W1 as W_DST
    W1 = W * H_DST / H # H1 as H_DST

    # prevent img cut
    if H1 > H_DST:
        H1 = H_DST
    elif W1 > W_DST:
        W1 = W_DST

    # new shape as int
    W1 = int(W1)
    H1 = int(H1)
    # resize img with keep ratio
    img = cv.resize(img,(W1,H1), interpolation=INTER)
    # final shape
    out_img = np.zeros((H_DST,W_DST,img.shape[2]), dtype=img.dtype)
    # paste resized img, img parts can have black areas
    out_img[:H1,:W1,:] = img

    return out_img, W1,H1


if __name__ == "__main__":
    gen = YOLODataGenerator('',[30,30],[],3)
    print(gen.__dict__.keys())
    
    def draw(img, bboxes,color):
        for bbox in bboxes:
            box = utils.xcycwh_to_ltrb(bbox.copy().reshape((-1,4))).astype(int).tolist()[0]
            l,t,r,b = box
            img = cv.rectangle(img,(l,t,),(r,b),color,1)
        return img
        
    X = cv.imread("img.jpg").astype(np.float32)/255
    X = cv.resize(X,(400,200))
    Y = np.array([
                  [10,10,20,20],
                  [389,189,20,20],
                  [389,10,20,20],
                  [389/2,189/2,50,20],
                  [389/3,189/3,10,80],
                  ],dtype=np.float32)
    
    X = draw(X.copy(),Y,(255,0,0))
    cv.imshow("ORIGINAL",X)
    print(Y)
    
    T = YOLOTransformer()
    
    X,Y = T.rotate(X,Y,45,45)
    X = draw(X.copy(),Y,(0,0,255))
    print(Y)
    cv.imshow("rotation",X)
    
    X,Y = T.fill(X,Y,0.1,0.6)
    X = draw(X.copy(),Y,(0,50,255))
    print(Y)
    cv.imshow("fill",X)
    
    X,Y = T.translate(X,Y,(-130,-129),(-145,-144))
    X = draw(X.copy(),Y,(200,50,200))
    print(Y)
    cv.imshow("translation",X)
    
    X,Y = T.horizontal_flip(X,Y)
    X = draw(X.copy(),Y,(0,255,255))
    print(Y)
    cv.imshow("horizontal flip",X)
    
    X,Y = T.vertical_flip(X,Y)
    X = draw(X.copy(),Y,(100,100,255))
    print(Y)
    cv.imshow("vertical flip",X)
    
    X,Y = T.scale(X,Y,0.6,0.6)
    X = draw(X.copy(),Y,(0,255,0))
    print(Y)
    cv.imshow("scale-",X)
    
    X,Y = T.scale(X,Y,1.5,1.5)
    X = draw(X.copy(),Y,(255,0,255))
    print(Y)
    cv.imshow("scale+",X)
    
    X,Y = T.YCrCb(X,Y,0.1)
    X = draw(X.copy(),Y,(255,0,255))
    print(Y)
    cv.imshow("ycrcb",X)
    
    X,Y = T.HSV(X,Y,0.1)
    X = draw(X.copy(),Y,(255,0,255))
    print(Y)
    cv.imshow("hsv",X)
    
    X,Y = T.blur(X,Y,7)
    X = draw(X.copy(),Y,(255,0,255))
    print(Y)
    cv.imshow("blur",X)
    
    X,Y = T.dilate(X,Y,7)
    X = draw(X.copy(),Y,(255,0,255))
    print(Y)
    cv.imshow("dilate",X)
    
    X,Y = T.erode(X,Y,7)
    X = draw(X.copy(),Y,(255,0,255))
    print(Y)
    cv.imshow("erode",X)
    
    X,Y = T.equalize_hist(X,Y,0.5)
    X = draw(X.copy(),Y,(255,0,255))
    print(Y)
    cv.imshow("eqhist",X)
    
    
    k = cv.waitKey(0)
    
    pass
