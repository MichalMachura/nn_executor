from functools import singledispatch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from ..utils import log_print



def seconds_to_hmsms(seconds):
    """
    Convert float number of seconds into string format h:min:s:ms
    """
    s = int(seconds)
    ms = int((seconds-s)*1000)
    min = s //60
    h = min // 60

    min -= h*60
    s -= h*60*60 + min*60

    txt = str(h)+'[h]:' if h else ''
    txt += str(min)+'[min]:' if min else ''
    txt += str(s)+'[s]:' if s else ''
    txt += str(ms)+'[ms]'

    return txt


def bar(value, max_value, pre_text='', post_text='', length=100, end='', completed_char='█', in_progress_char='>', to_process_char=' '):

	state = int(float(value)/max_value*length)
	rest = length - state

	s = '\r'+ pre_text + ' [' + completed_char*state

	if rest > 0:
		s += in_progress_char
		rest -= 1

	s += to_process_char*rest
	s += '] '
	s += '['+str(value)+'/'+str(max_value)+'] '

	s += post_text

	log_print(s, end=end)


def draw_bbox(img, bbox, color=(255,0,0), thickness=1):
	"""
	Draw bounding box on image img with given color and line's thickness
	:param img: image np.array
	:param bbox: bounding box np.array([xc,yc,w,h])
	:param color: tuple(RGB) or nuber value for intensity image
	:param thickness: line size
	:return: img with drawn bbox
	"""
	xmin,xmax,ymin,ymax =  bbox[0]-bbox[2]/2,bbox[0]+bbox[2]/2,bbox[1]-bbox[3]/2,bbox[1]+bbox[3]/2
	img = cv.rectangle(img,(int(xmin), int(ymin)), (int(xmax), int(ymax)),color, thickness)

	return img


def unravel_index(index, shape):
    """
    """
    out = []
    for dim in reversed(shape):
        idx = index % dim
        out.append(idx)
        index //= dim

    return tuple(reversed(out))


@singledispatch
def xcycwh_to_ltrb(bbox_batch):
    raise NotImplementedError()

@xcycwh_to_ltrb.register
def _torch(bbox_batch: torch.Tensor) -> torch.Tensor:
    # move center to left top
    LT = bbox_batch[:,:2] - bbox_batch[:,-2:]/2
    # change wh into right bottom
    RB = LT + bbox_batch[:,-2:]
    LTRB = torch.cat([LT, RB], dim=1)
    return LTRB

@xcycwh_to_ltrb.register
def _numpy(bbox_batch: np.ndarray) -> np.ndarray:
    # move center to left top
    LT = bbox_batch[:,:2] - bbox_batch[:,-2:]/2
    # change wh into right bottom
    RB = LT + bbox_batch[:,-2:]
    LTRB = np.concatenate()([LT, RB], axis=1)
    return LTRB


@singledispatch
def ltrb_to_xcycwh(bbox_batch):
    raise NotImplementedError()

@ltrb_to_xcycwh.register
def _torch(bbox_batch: torch.Tensor) -> torch.Tensor:
    # change right bottom into wh
    WH = bbox_batch[:,-2:] - bbox_batch[:,:2]
    # move left top to center
    bbox_batch[:,:2] += bbox_batch[:,-2:] / 2
    XcYc = (bbox_batch[:,-2:] + bbox_batch[:,:2]) / 2
    XcYcWH = torch.cat([XcYc, WH], dim=1)
    return XcYcWH

@ltrb_to_xcycwh.register
def _numpy(bbox_batch: np.ndarray) -> np.ndarray:
    # change right bottom into wh
    WH = bbox_batch[:,-2:] - bbox_batch[:,:2]
    # move left top to center
    bbox_batch[:,:2] += bbox_batch[:,-2:] / 2
    XcYc = (bbox_batch[:,-2:] + bbox_batch[:,:2]) / 2
    XcYcWH = np.concatenate([XcYc, WH], axis=1)
    return XcYcWH


def plot_history(hist, formatable_path:str=None):
    """
    :param hist: dict like key : list of values
                 where:
                 - key is name of loss, metric, etc.
                 list of values is list of values obtained at the whole training process
    :formatable_path: formatable path, which allow to format with one arg. as key name
                 to save it's history into file
    """
    keys = [k for k in hist.keys() if 'val_' not in k]

    for k in keys:
        v1 = hist[k]
        v2 = hist['val_'+k]

        v1 = [v.item() if isinstance(v, torch.Tensor) else v for v in v1]
        v2 = [v.item() if isinstance(v, torch.Tensor) else v for v in v2]

        size = min(len(v1),len(v2))
        x = np.arange(0,size)+1

        plt.plot(x, v1[:size], label=k)
        plt.plot(x, v2[:size], label='val_'+k)
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(k+' value')
        plt.title("History of '{}'".format(k))

        if formatable_path is not None:
            path = formatable_path.format(k)
            plt.savefig(path)

        plt.show()

