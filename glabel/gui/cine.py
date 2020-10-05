import struct
import numpy as np

UINT8_T = "B"
CHAR = "c"
UINT16_T = "H"
INT16_T = "h"
BOOL32_T = "I"
UINT32_T = "I"
INT32_T = "i"
INT64_T = "q"
FLOAT = "f"
DOUBLE = "d"
CHAR_ARR = "s"

def unpack(h, fmt="i"):
    """Quick unpack from bytes"""
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, h[:size]), h[size:]

class tagCINEFILEHEADER(object):
    """CINE file header"""
    def __init__(self):
        pass
    
    def __len__(self):
        return 44
    
    def build(self, h):
        self.Type, h = unpack(h, CHAR+CHAR)
        self.Type = self.Type[0]+self.Type[1]
        
        self.Headersize, h = unpack(h, UINT16_T)
        self.Compression, h = unpack(h, UINT16_T)
        self.Version, h = unpack(h, UINT16_T)
        self.FirstMovieImage, h = unpack(h, INT32_T)
        self.TotalImageCount, h= unpack(h, UINT32_T)
        self.FirstImageNo, h = unpack(h, INT32_T)
        self.ImageCount, h = unpack(h, UINT32_T)
        self.OffImageHeader, h = unpack(h, UINT32_T)
        self.OffSetup, h = unpack(h, UINT32_T)
        self.OffImageOffsets, h = unpack(h, UINT32_T)
        self.TriggerTime, h = unpack(h, UINT32_T+UINT32_T)

class tagBITMAPINFOHEADER:
    """Bitmap header"""
    def __len__(self):
        return 40
    def __init__(self):
        pass
    
    def build(self, h):
        self.biSize, h = unpack(h, UINT32_T)
        self.biWidth, h = unpack(h, INT32_T)
        self.biHeight, h = unpack(h, INT32_T)
        self.biPlanes, h = unpack(h, UINT16_T)
        self.biBitCount, h = unpack(h, UINT16_T)
        self.biCompression, h = unpack(h, UINT32_T)
        self.biSizeImage, h = unpack(h, UINT32_T)
        self.biXPelsPerMeter, h = unpack(h, INT32_T)
        self.biYPelsPerMeter, h = unpack(h, INT32_T)
        self.biClrUsed, h = unpack(h, UINT32_T)
        self.biClrImportant, h = unpack(h, UINT32_T)
        
    def show(self):
        print("biSize :", self.biSize)
        print("biWidth :", self.biWidth)
        print("biHeight :", self.biHeight)
        print("biPlanes :", self.biPlanes)
        print("biBitCount :", self.biBitCount)
        print("biCompression :", self.biCompression)
        print("biSizeImage :", self.biSizeImage)
        print("biXPelsPerMeter :", self.biXPelsPerMeter)
        print("biYPelsPerMeter :", self.biYPelsPerMeter)
        print("biClrUsed :", self.biClrUsed)
        print("biClrImportant :", self.biClrImportant)



def conv_10_bit(byte_buf, height, width) -> np.ndarray:
    """Creates image from 10 bit packed in 5 bytes in uint16 numpy array.
    Taken from https://stackoverflow.com/questions/38302765/python-reading-10-bit-integers-from-a-binary-file

    Parameters
    ----------
    byte_buf : bytearray
        image data
    height : int
        image height
    width : int
        image width

    Returns
    -------
    numpy.ndarray
        Image (uin16, HxW)
    """
    data = np.frombuffer(byte_buf, dtype=np.uint8)
    # 5 bytes contain 4 10-bit pixels (5x8 == 4x10)
    b1, b2, b3, b4, b5 = np.reshape(data, (data.shape[0]//5, 5)).astype(np.uint16).T
    o1 = (b1 << 2) + (b2 >> 6)
    o2 = ((b2 % 64) << 4) + (b3 >> 4)
    o3 = ((b3 % 16) << 6) + (b4 >> 2)
    o4 = ((b4 % 4) << 8) + b5

    unpacked =  np.reshape(np.concatenate((o1[:, None], o2[:, None], o3[:, None], o4[:, None]), axis=1), (height, width))
    return unpacked


class Image:
    """Creates an image from raw byte stream"""
    def __init__(self, h, width, height, bits=10):
        annotation_size, h = unpack(h, UINT32_T)
        annotation_size = annotation_size[0]
        
        # Remove annotation header (4 bytes for annot. length)
        h = h[annotation_size-4:]
        
        self.width = width
        self.height = height
        self.bits = bits
        self.src = self._build(h)
        
    def _build(self, h):
        """Builds image from data

        Parameters
        ----------
        h : bytearray
            Image data

        Returns
        -------
        numpy.ndarray or False
            Provides Image (HxW) on success.
        """
        if self.bits == 10:
            return conv_10_bit(h, self.height, self.width)
        
        elif self.bits == 8:
            return np.frombuffer(h,
                        dtype=np.uint8).reshape(self.height, self.width)
        
        else:
            print("Not implemented...")
            return None