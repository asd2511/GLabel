import glob
import re

import numpy as np


def parse_yolo_log(log_file, cfg_file=None, total_samples=None):
    batch_lines = []
    map_lines = [[1, '0.00000'],]  # Initialize mAP and IoU values with 0% at
    iou_lines = [[1, '00.00'], ]   # training start
    with open(log_file, 'r') as f:
        for line in f:
            # Lines starting as ' (batch number): ...' are the lines at the beginning of a new
            # batch. These lines contain the values for losses, learning rate, timings, ...
            if re.match(r'^\s\d*:\s', line):
                batch_num = line.split(':')[0].strip()
                batch_lines.append([int(batch_num), line.rstrip()])
            
            # Lines containing the string 'mean_average_precision' list the mAP value between
            # batches
            if 'mean_average_precision' in line:
                # batch_num = batch_lines[-1].split(':')[0].strip()
                map_lines.append([int(batch_num), line.rstrip()])

            # Lines containg the string 'average IoU' list the average IoU using the validation
            # set
            if 'average IoU' in line:
                # batch_num = batch_lines[-1].split(':')[0].strip()
                iou_lines.append([int(batch_num), line.rstrip()])

    # The first value in the batch lines is the current loss, the second one the average loss that
    # we are interested in
    avg_losses = [[batch[0], float(re.findall(r'\s\d*\.\d*', batch[1])[1])] for batch in batch_lines]
    
    # The mAP lines contain the mean average precision value as the only value written with a precision
    # higher than 2 decimal places
    maps = [[mapline[0], float(re.findall(r'\d\.\d{3,}', mapline[1])[0])] for mapline in map_lines]

    # The IoU lines contain the average IoU value as the last percentage value of the line
    # We search for the last value with an '=' sign and remove the last to characters (' %')
    ious = [[iouline[0], float(iouline[1].split(' = ')[-1][:-2])/100.0] for iouline in iou_lines]

    return avg_losses, maps, ious


def stitch_yolo_logs(log_dir, total_batches):
    log_files = glob.glob(log_dir + '/*.txt')
    stitched_losses = np.empty(total_batches)
    stitched_losses[:] = np.nan
    stitched_maps = np.empty(total_batches)
    stitched_maps[:] = np.nan
    stitched_ious = np.empty(total_batches)
    stitched_ious[:] = np.nan

    for file in log_files:
        losses, maps, ious = parse_yolo_log(file)
        for loss in losses:
            stitched_losses[loss[0]-1] = loss[1]
        for i_map in maps:
            stitched_maps[i_map[0]-1] = i_map[1]
        for iou in ious:
            stitched_ious[iou[0]-1] = iou[1]
    stitched_maps[499] = np.nan  # Delete the validation after batch 500 to make plots uniform with Tiny YOLOv3 training
    stitched_ious[499] = np.nan  # where no validation took place at 500.

    batch_idcs = np.arange(1, total_batches+1)

    masked_losses = np.ma.masked_invalid(stitched_losses)
    loss_idcs = batch_idcs[~masked_losses.mask]

    masked_maps = np.ma.masked_invalid(stitched_maps)
    map_idcs = batch_idcs[~masked_maps.mask]

    masked_ious = np.ma.masked_invalid(stitched_ious)
    iou_idcs = batch_idcs[~masked_ious.mask]

    stitched_losses_list = [[idx, val] for idx, val in zip(loss_idcs, masked_losses.compressed())]
    stitched_maps_list = [[idx, val] for idx, val in zip(map_idcs, masked_maps.compressed())]
    stitched_ious_list = [[idx, val] for idx, val in zip(iou_idcs, masked_ious.compressed())]

    return stitched_losses_list, stitched_maps_list, stitched_ious_list
