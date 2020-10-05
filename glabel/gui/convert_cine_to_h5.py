import struct
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QThreadPool, Qt
from joblib import Parallel, delayed
import numpy as np
import flammkuchen as fl 
from PyQt5.QtWidgets import QFileDialog, QApplication, QDialog, QPushButton, QVBoxLayout, QProgressDialog

from glabel.gui import worker
from glabel.gui.cine import tagCINEFILEHEADER, tagBITMAPINFOHEADER, INT64_T, Image

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


class ModeDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.mode = None

        self.layout = QVBoxLayout()

        self.file_btn = QPushButton("Convert single File")
        self.file_btn.clicked.connect(lambda: self.set_mode('path'))
        self.dir_btn = QPushButton("Convert all in Directory")
        self.dir_btn.clicked.connect(lambda: self.set_mode('dir'))
        # TODO: Implement batch conversion
        self.dir_btn.setEnabled(False)

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
                if prog_dlg.wasCanceled():
                    return
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
        if prog_dlg:
            prog_dlg.setLabelText("Saving converted image data...")
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
    thread_worker.signals.finished.connect(lambda: prog_dlg.setValue(prog_dlg.maximum()))

    mw.threadpool.start(thread_worker)


if __name__ == '__main__':
    app = QApplication([])
    start_converter()
    QCoreApplication.quit()
