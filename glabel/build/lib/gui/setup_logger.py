#!/usr/bin/env python
import os
import logging

loggers = []
main_logger = logging.getLogger('main')
loggers.append(main_logger)
image_stack_logger = logging.getLogger('img_stack')
loggers.append(image_stack_logger)
grid_widget_logger = logging.getLogger('grid_widget')
loggers.append(grid_widget_logger)
img_view_logger = logging.getLogger('img_view')
loggers.append(img_view_logger)
roi_logger = logging.getLogger('roi')
loggers.append(roi_logger)

log_dir = '../.logs/'
try:
    os.mkdir(log_dir)
except FileExistsError:
    pass

fh = logging.FileHandler(log_dir + 'suture_lab.log', 'w')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

for logger in loggers:
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
