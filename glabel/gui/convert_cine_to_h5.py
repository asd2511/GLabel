import struct
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QThreadPool, Qt
from joblib import Parallel, delayed
import numpy as np
import flammkuchen as fl 
from PyQt5.QtWidgets import QFileDialog, QApplication, QDialog, QPushButton, QVBoxLayout, QProgressDialog

from glabel.gui import worker

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
    """CINE path header"""
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


def conv_10_bit(im_raw):
    ''' Generate 10 bit (unsigned) integers from a binary source
    
    im_raw: 
        bytes
        
    return:
        list of (unsigned) integers'''
    
    im = []
    
    for i in range(0, len(im_raw), 5):
        b = im_raw[i:i+5]
        n = int.from_bytes(b, 'big')

        #Split n into 4 10 bit integers
        t = []
        
        for i in range(4):
            t.append(n & 0x3ff)
            # Bitshift by 10 bits
            n >>= 10
            
        # Reverse bits, otherwise it is corrupted (little vs big endian)
        im.extend(reversed(t))
        
    return im


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
        self.src = self.build(h)
        
    def build(self, h):
        if self.bits == 10:
            return np.asarray(conv_10_bit(h), 
                        dtype=np.uint16).reshape(self.height, self.width)
        
        elif self.bits == 8:
            return np.frombuffer(h,
                        dtype=np.uint8).reshape(self.height, self.width)
        
        else:
            print("Not implemented...")
            return None


class ModeDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.mode = None

        self.layout = QVBoxLayout()

        self.file_btn = QPushButton("Convert single File")
        self.file_btn.clicked.connect(lambda: self.set_mode('path'))
        self.dir_btn = QPushButton("Convert all in Directory")
        self.dir_btn.clicked.connect(lambda: self.set_mode('dir'))

        self.layout.addWidget(self.file_btn)
        self.layout.addWidget(self.dir_btn)
        self.setLayout(self.layout)

    def set_mode(self, mode):
        self.mode = mode
        self.accept()


def convert_file(filename, prog_dlg=None, prog_clb=None, msg_clb=None, cnc_clb=None):
    print(f"Converting {filename}... to {filename[:-5]}.h5")
    if filename:
        with open(filename, "rb") as fp:
            h = fp.read()

        header = tagCINEFILEHEADER()
        header.build(h[:len(header)])

        print("Total Images: ", header.TotalImageCount[0])
        if prog_dlg:
            prog_dlg.setMaximum(header.TotalImageCount[0])
            prog_dlg.setLabelText(f"0 / {header.TotalImageCount[0]}")

        imageHeader = tagBITMAPINFOHEADER()
        imageHeader.build(h[len(header):len(header)+len(imageHeader)])

        imageHeader.show()

        # Not nice, but functional; each offset is a INT64_T number
        raw_im_offsets = h[header.OffImageOffsets[0]:header.OffImageOffsets[0]+header.TotalImageCount[0]*8]
        # Convert to real int index offset
        im_offsets = [struct.unpack(INT64_T, raw_im_offsets[i:i+8])[0] for i in range(0, header.TotalImageCount[0]*8, 8)]

        real_ims = []

        # Iterate over offsets
        for count, i in enumerate(im_offsets):
            if prog_dlg:
                prog_dlg.setValue(count)
                prog_dlg.setLabelText(f"{count} / {len(im_offsets)}")

            if count % 500 == 0:
                print(f"\nFrame_count {count} for {filename}!\n")
            # Currently +8 as magic number and not inferred (Annotation size)
            raw_im = h[i:i+imageHeader.biSizeImage[0]+8]
            real_im = Image(raw_im,
                            imageHeader.biWidth[0],
                            imageHeader.biHeight[0]).src

            real_ims.append(real_im)

        print(f"Conversion of {filename} finished. Preparing to save...")

        real_ims = np.asarray(real_ims, dtype=np.uint16)

        # Save as hdf5
        print("Save real images as hdf5...")
        fl.save(filename[:-5]+".h5", dict(ims=real_ims), compression=('blosc', 9))
        print("Done!")

        # plot = pg.image(real_ims.transpose(0, 2, 1))


def start_converter():
    mode = None
    mode_dlg = ModeDialog()
    if mode_dlg.exec_():
        mode = mode_dlg.mode
    else:
        exit(0)

    if mode == 'dir':
        path = QFileDialog.getExistingDirectory()
        files = list(Path(path).rglob('*.cine'))
        print(f"Converting files: {files}")
        Parallel(n_jobs=4)(delayed(convert_file)(str(file)) for file in files)
    elif mode == 'path':
        file = QFileDialog.getOpenFileName()[0]
        print(f"Converting file: {file}")
        convert_file(file)


def ask_files():
    mode = None
    mode_dlg = ModeDialog()
    if mode_dlg.exec_():
        mode = mode_dlg.mode
    else:
        exit(0)

    if mode == 'dir':
        path = QFileDialog.getExistingDirectory()
        files = list(Path(path).rglob('*.cine'))
    elif mode == 'path':
        files = QFileDialog.getOpenFileName()[0]
    else:
        raise ValueError(f"Invalid value {mode} for `mode` encountered in `ask_files`!")

    return files


def run_as_modal(mw):
    files = ask_files()

    prog_dlg = QProgressDialog("Converting .CINE file...", "Abort", 0, 2000, mw, Qt.Dialog)
    prog_dlg.setWindowTitle("Converter")
    prog_dlg.setWindowModality(Qt.WindowModal)
    prog_dlg.setValue(0)
    prog_dlg.show()

    thread_worker = worker.Worker(convert_file, files, prog_dlg)

    mw.threadpool.start(thread_worker)


if __name__ == '__main__':
    app = QApplication([])
    start_converter()
    QCoreApplication.quit()
