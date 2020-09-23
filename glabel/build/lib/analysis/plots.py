import json
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
import seaborn as sns

from glabel.analysis import utils, evaluation_utils
from glabel.nn import nn_utils

# sns.set_style('whitegrid')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
# thesis_palette = (
#     (0/256, 177/256, 235/256, 1),
#     (0/256, 163/256, 121/256, 1)
# )

SAVEDIR = r'../../figures'


def set_thesis_settings():
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (7, 3.5)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['axes.titlesize'] = 12.0
    plt.rcParams['axes.labelsize'] = 12.0
    plt.rcParams['xtick.labelsize'] = 8.0
    plt.rcParams['ytick.labelsize'] = 8.0
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['hatch.linewidth'] = 0.25


def percentage_countplot(*args, **kwargs):
    try:
        data = kwargs['data']
    except KeyError as e1:
        try:
            data = args[0]
        except IndexError as e2:
            raise(e1)
            raise(e2)

    ncount = len(data)
    ax = sns.countplot(*args, **kwargs)

    # Make twin axis
    ax2 = ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Switch labels too
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel('Frequency [%]')

    # Add percentages above boxes
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:.2f}%'.format(100. * y/ncount), (x.mean(), y), ha='center', va='bottom')

    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))

    # Fix the frequency range to [0, 100]
    ax2.set_ylim(0, 100)
    ax.set_ylim(0, ncount)

    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

    ax2.set_zorder(-1)

    return ax, ax2


def sortnet_input_example(file, frame, side, highlight_id, peak_r, blob=False, blob_sigma=1.0, return_objects=False):
    rois = utils.load_rois_patch_relative(file, (224, 224))
    rois = rois[frame]
    rois = rois[:35] if side == 0 else rois[35:]

    patch = utils.extract_rois_patch(file, 0, (224, 224), True)
    patch = patch[frame*2 + side]

    if not blob:
        peak_fill = np.ones((peak_r*2+1, peak_r*2+1))
    else:
        x, y = np.meshgrid(np.linspace(-1, 1, peak_r*2+1), np.linspace(-1, 1, peak_r*2+1))
        d = np.sqrt(x * x + y * y)
        sigma, mu = blob_sigma, 0.0
        peak_fill = (np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))))

    input_map = np.zeros((224, 224, 3))
    for idx, roi in enumerate(rois):
        if not any(coord == -1 for coord in roi):
            y = np.round(roi[1] * 224).astype(int)
            x = np.round(roi[0] * 224).astype(int)
            input_map[y-peak_r-1:y+peak_r, x-peak_r-1:x+peak_r, 0] = peak_fill
            if idx == highlight_id:
                input_map[y-peak_r-1:y+peak_r, x-peak_r-1:x+peak_r, 1] = peak_fill
                input_map[y-peak_r-1:y+peak_r, x-peak_r-1:x+peak_r, 2] = peak_fill

    fig1, axes1 = plt.subplots(1, 2)
    axes1[0].imshow(patch, cmap='gray')
    axes1[0].axis('off')

    axes1[1].imshow(input_map)
    axes1[1].axis('off')

    if not return_objects:
        fig1.show()

    fig2, axes2 = plt.subplots(1, 3)
    for i in range(3):
        single_channel = np.zeros((224, 224, 3))
        single_channel[:, :, i] = input_map[:, :, i]
        axes2[i].imshow(single_channel)
        axes2[i].axis('off')
    if not return_objects:
        fig2.show()
    else:
        return fig1, axes1, fig2, axes2


def total_evaluation_confusion(df, return_objects=False):
    conf_mat = evaluation_utils.get_placement_confusion(df['placed_conf'], normalize=True)

    ax = sns.heatmap(conf_mat, annot=True, cmap='viridis', xticklabels=['Placed', 'Missing'],
                     yticklabels=['Placed', 'Missing'], square=True)
    ax.set_ylim([2, 0])
    ax.set_title('Total')
    ax.set_ylabel('GT')
    ax.set_xlabel('Prediction')

    if return_objects:
        return ax


def confusion_matrix(x, **kwargs):
    conf_mat = evaluation_utils.get_placement_confusion(x, normalize=True)
    sns.heatmap(conf_mat, annot=True, cmap='viridis', xticklabels=['Placed', 'Missing'],
                yticklabels=['Placed', 'Missing'], **kwargs)
    plt.ylim([2, 0])
    plt.ylabel('GT')
    plt.xlabel('Prediction')


def file_split_confusion(df):
    g = sns.FacetGrid(df, col='file_id', height=4, aspect=1.5, gridspec_kws={'wspace': .3})
    g = g.map(confusion_matrix, "placed_conf")

    for i in range(3):
        fname = os.path.basename(df[df['file_id'] == i]['file'].iloc[0])
        parts = fname.split('_')
        fname = '_'.join([parts[1], parts[-3], parts[-2]])
        g.axes[0, i].set_title(fname)
        g.axes[0, i].set_xlabel('Prediction')
        if i == 0:
            g.axes[0, i].set_ylabel('GT')

    return g


def show_single_suture(file, frame, side, row, col, size, show=False):
    side_offset = 0 if side in ['l', 'left'] else 1
    idx = frame//2 + side_offset
    patch = utils.extract_rois_patch(file, 0, (224, 224), True)[idx]
    rois = utils.load_rois_patch_relative(file, (224, 224))[idx]
    side_rois = rois[:35] if side in ['l', 'left'] else rois[35:]
    exact_roi = side_rois[row*5 + col]
    y = int(exact_roi[1] * 224)
    x = int(exact_roi[0] * 224)
    h = size[0] // 2
    w = size[1] // 2

    roi_region = patch[y-h:y+h+1, x-w:x+w+1]

    plt.imshow(roi_region, cmap='gray', vmin=0.0, vmax=255.0)
    plt.axis('off')


def plot_yolo_history(log_file, thesis=False, batchrange=None):
    if thesis:
        set_thesis_settings()
        plt.rcParams['figure.figsize'] = (3.2, 2.2)

    if os.path.isfile(log_file):
        loss, maps, ious = nn_utils.parse_yolo_log(log_file, '', 0)
    else:
        loss, maps, ious = nn_utils.stitch_yolo_logs(log_file, 6000)
    loss = np.asarray(loss)
    maps = np.asarray(maps)
    ious = np.asarray(ious)

    if batchrange is not None:
        loss = loss[batchrange[0]:batchrange[1]]
        maps = maps[np.where((batchrange[0] <= maps[:, 0]) & (maps[:, 0] <= batchrange[1]))]
        ious = ious[np.where((batchrange[0] <= ious[:, 0]) & (ious[:, 0] <= batchrange[1]))]

    fig, ax = plt.subplots()

    x = np.arange(len(loss)) if batchrange is None else np.linspace(batchrange[0], batchrange[1], len(loss))

    # p1 = sns.lineplot(x=x, y=loss, ax=ax, label='Avg. Loss')
    p1 = sns.lineplot(x=loss[:, 0], y=loss[:, 1], ax=ax, label='Avg. Loss')
    ax.set_xlabel('Training Batch')
    ax.set_ylabel('Avg. Loss')
    ax.grid(False)
    ax.get_legend().remove()
    ax.set_yscale('log')
    ax.set_ylim([0.009, 9000])

    lax = ax.twinx()
    p2 = sns.lineplot(x=maps[:, 0], y=maps[:, 1], ax=lax, color='r', label='mAP')
    p3 = sns.lineplot(x=ious[:, 0], y=ious[:, 1], ax=lax, color='g', label='Avg. IoU')
    lax.grid(False)
    lax.set_ylabel('mAP\nAvg. IoU')
    lax.get_legend().remove()

    # Create one legend listing all curves
    lines = ax.lines + lax.lines
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc=0, bbox_to_anchor=(1.6, 1.2), frameon=False)

    sns.despine(right=False)


def plot_yolo_iou_comp(df) -> None:
    """
    Plot comparison of suture grid region IoU between YOLOv3 and Tiny YOLOv3 network.

    The passed DataFrame must have the same columns and formatting as created by the
    `yolo_evaluation.ipynb` notebook.

    :param pd.DataFrame df: Evaluation DataFrame containing data on IoU of networks.
    """
    set_thesis_settings()
    plt.rcParams['figure.figsize'] = (6/2.54, 6/2.54)

    long_df = df.melt(
        id_vars=['frame', 'side', 'file'],
        value_vars=['yolo IoU', 'tiny IoU'],
        var_name='Network',
        value_name='IoU'
    )
    long_df['Network'].replace(
        ['yolo IoU', 'tiny IoU'],
        ['YOLOv3', 'Tiny YOLOv3'],
        inplace=True
    )
    long_df['side'] = long_df['side'].str.capitalize()
    long_df.rename(columns={'side': 'Side'}, inplace=True)

    sns.boxplot(
        y='IoU',
        x='Network',
        data=long_df,
        width=.5,
        palette='tab10'
    )
    sns.despine()
    plt.xlabel('')

    print(long_df.groupby('Network')['IoU'].describe())


def plot_yolo_time_comp(df) -> None:
    """
    Plot comparison of suture grid region inference time between YOLOv3 and Tiny YOLOv3
    network.

    The passed DataFrame must have the same columns and formatting as created by the
    `yolo_evaluation.ipynb` notebook.

    :param pd.DataFrame df: Evaluation DataFrame containing data on inference times of
        networks.
    """
    set_thesis_settings()
    plt.rcParams['figure.figsize'] = (6/2.54, 6/2.54)

    long_df = df.melt(
        id_vars=['frame', 'side', 'file'],
        value_vars=['yolo times','tiny times'],
        var_name='Network',
        value_name='Inference time [ms]'
    )
    long_df['Network'].replace(
        ['yolo times', 'tiny times'],
        ['YOLOv3', 'Tiny YOLOv3'],
        inplace=True
    )
    long_df['Inference time [ms]'] *= 1000  # Scale to [ms]

    sns.boxplot(
        x='Network',
        y='Inference time [ms]',
        data=long_df,
        width=.5,
        palette='tab10'
    )
    sns.despine()
    plt.xlabel('')

    print(long_df.groupby('Network')['Inference time [ms]'].describe())


def plot_yolo_center_comp(df) -> None:
    """
    Plot comparison of suture grid bounding box center offsets between YOLOv3 and
    Tiny YOLOv3.

    The center offset of a bounding box is the vertical and horizontal distance to
    the ground truth bounding box. This distance is of relevance, because the suture
    segmentation expands these bounding boxes and the complete suture grid should be
    contained in these expanded boundaries.

    The passed DataFrame must have the same columns and formatting as created by the
    `yolo_evaluation.ipynb` notebook.

    :param pd.DataFrame df: Evaluation DataFrame containing data on inference times of
        networks
    """
    set_thesis_settings()
    plt.rcParams['figure.figsize'] = (3.5, 3.)

    long_df = df.melt(
        id_vars=['frame', 'side', 'file'],
        value_vars=['yolo dx', 'yolo dy', 'tiny dx', 'tiny dy'],
        var_name='to split',
        value_name='Offset'
    )
    long_df[['Network', 'Direction']] = long_df['to split'].str.split(" ", n=1, expand=True)
    long_df['Network'].replace(
        ['yolo', 'tiny'],
        ['YOLOv3', 'Tiny YOLOv3'],
        inplace=True
    )

    # for net in ['YOLOv3', 'Tiny YOLOv3']:
    #     sns.kdeplot(
    #         long_df['Offset'][(long_df['Network'] == net) & (long_df['Direction'] == 'dx')],
    #         long_df['Offset'][(long_df['Network'] == net) & (long_df['Direction'] == 'dy')],
    #         # shade=True,
    #         # shade_lowest=False,
    #         label=net,
    #         # alpha=.5
    #     )
    # plt.xlabel('Horizontal offset [px]')
    # plt.ylabel('Vertical offset [px]')

    # ax = plt.gca()
    # max_lim = max(np.abs([*ax.get_ylim(), *ax.get_xlim()]))
    # ax.set_ylim([-max_lim, max_lim])
    # ax.set_xlim([-max_lim, max_lim])

    # -----

    fig, axes = plt.subplots(1, 2, figsize=(14/2.54, 6/2.54))

    sns.boxplot(
        x='Network',
        y='Offset',
        data=long_df[long_df['Direction'] == 'dy'],
        ax=axes[0],
        width=.5,
        palette='tab10'
    )
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Verical offset [px]')


    sns.boxplot(
        y='Network',
        x='Offset',
        data=long_df[long_df['Direction'] == 'dx'],
        orient='h',
        ax=axes[1],
        width=.5,
        palette='tab10'
    )
    axes[1].yaxis.tick_right()
    axes[1].set_ylabel('')
    axes[1].set_xlabel('Horizontal offset [px]')

    sns.despine()

    # -----

    # sns.violinplot(
    #     x='Network',
    #     y='Offset',
    #     hue='Direction',
    #     split=True,
    #     data=long_df
    # )

    print(long_df.groupby(['Network', 'Direction'])['Offset'].describe())


def plot_peak_find_comp(patch, map_a, map_b, peaks_a, peaks_b, gt_peaks):
    set_thesis_settings()

    patch_cmap = 'gray'
    map_cmap = 'magma'
    map_cmap = plt.cm.get_cmap(map_cmap)
    highlight_range = {'x': [80, 120], 'y': [100, 60]}
    map_alpha = 0.99

    masked_a = np.ma.masked_where(map_a < 0.0, map_a)  # Masking currently not used to make clear that overlay of
    masked_b = np.ma.masked_where(map_b < 0.0, map_b)  # probability map is shown.

    pred_marker = dict(facecolor='none', edgecolor='deepskyblue', lw=3,  s=15**2)
    gt_marker = dict(facecolor='none', edgecolor='magenta', lw=3, s=15**2)

    fig, axes = plt.subplots(2, 3, figsize=(6, 4),
                             gridspec_kw=dict(wspace=.1, hspace=.1))

    # Original input patch
    axes[0, 0].imshow(patch, cmap=patch_cmap)
    axes[0, 0].axis('off')
    axes[0, 0].grid(False)
    axes[0, 0].set_title('Input image')

    # Highlight region of originol patch
    axes[1, 0].imshow(patch, cmap=patch_cmap)
    axes[1, 0].scatter(gt_peaks[:, 0], gt_peaks[:, 1], **gt_marker)
    axes[1, 0].set_xlim(highlight_range['x'])
    axes[1, 0].set_ylim(highlight_range['y'])
    axes[1, 0].axis('off')
    axes[1, 0].grid(False)

    axes[0, 1].imshow(patch, cmap=patch_cmap)
    axes[0, 1].imshow(masked_a, cmap=map_cmap, interpolation='none', alpha=map_alpha)
    axes[0, 1].axis('off')
    axes[0, 1].grid(False)
    axes[0, 1].set_title('Raw probability map')

    axes[1, 1].imshow(patch, cmap=patch_cmap)
    axes[1, 1].imshow(masked_a, cmap=map_cmap, interpolation='none', alpha=map_alpha)
    axes[1, 1].scatter(peaks_a[:, 1], peaks_a[:, 0], **pred_marker)
    # axes[1, 1].scatter(gt_peaks[:, 0], gt_peaks[:, 1], **gt_marker)
    axes[1, 1].set_xlim(highlight_range['x'])
    axes[1, 1].set_ylim(highlight_range['y'])
    axes[1, 1].axis('off')
    axes[1, 1].grid(False)

    axes[0, 2].imshow(patch, cmap=patch_cmap)
    im = axes[0, 2].imshow(masked_b, cmap=map_cmap, interpolation='none', alpha=map_alpha)
    axes[0, 2].axis('off')
    axes[0, 2].grid(False)
    axes[0, 2].set_title('Eroded probability map')

    axes[1, 2].imshow(patch, cmap=patch_cmap)
    axes[1, 2].imshow(masked_b, cmap=map_cmap, interpolation='none', alpha=map_alpha)
    axes[1, 2].scatter(peaks_b[:, 1], peaks_b[:, 0], **pred_marker)
    # axes[1, 2].scatter(gt_peaks[:, 0], gt_peaks[:, 1], **gt_marker)
    axes[1, 2].set_xlim(highlight_range['x'])
    axes[1, 2].set_ylim(highlight_range['y'])
    axes[1, 2].axis('off')
    axes[1, 2].grid(False)

    for col in range(3):
        axes[0, col].add_patch(Rectangle(
            xy=(highlight_range['x'][0], highlight_range['y'][0]),
            width=highlight_range['x'][1]-highlight_range['x'][0],
            height=highlight_range['y'][1]-highlight_range['y'][0],
            fill=False, color='magenta', lw=1, ls=':'
        ))

    br_bbox = axes[1, 2].get_position()
    l = br_bbox.xmax * 1.02
    b = br_bbox.ymin
    h = axes[0, 2].get_position().ymax - b
    cax = fig.add_axes((l, b, 0.01, h))
    fig.colorbar(im, cax=cax, orientation='vertical')


def plot_peak_find_misses(eval_file, method='both', acc=False):
    """
    Plotting comparison of missed sutures during peak finding between using raw probability maps and applying
    erosion operation beforehand.

    The passed .json file must be formatted as done by `peak_finding_toogoodnet.ipynb` notebook.

    :param eval_file: Path to .json file with data on missed annotations per file split
        between using raw or eroded probability maps.
    :param str method: ['both', 'erosion', 'edt'] For which type of probability map processing the comparison should be
        plotted.
    :param bool acc: Show accuracy instead of missing percentage. Accuracy is is calculated as (GT - missed) / GT.
    """
    from scipy.stats import normaltest, mannwhitneyu, ttest_ind

    set_thesis_settings()
    plt.rcParams['figure.figsize'] = (4, 3.5)

    with open(eval_file, 'r') as f:
        data = json.load(f)

    df_data = []
    for f_idx, filename in enumerate(list(data.keys())):
        for frame_idx, (mcom, mero, medt, gt) in enumerate(
                zip(
                    data[filename]['missed_center'],
                    data[filename]['missed_eroded'],
                    data[filename]['missed_edt'],
                    data[filename]['num_gt'])):
            df_data.append({
                'file index': f_idx,
                'frame': frame_idx,
                'Raw': mcom,
                'Eroded': mero,
                'EDT': medt,
                'GT': gt
            })

    if method == 'both':
        map_vars = ['Raw', 'Eroded', 'EDT']
    elif method == 'erosion':
        map_vars = ['Raw', 'Eroded']
    elif method == 'edt':
        map_vars = ['Raw', 'EDT']
    else:
        raise ValueError('Invalid value passed for `method`.')

    df = pd.DataFrame(df_data)
    df = df.groupby(['file index']).sum().reset_index()
    long_df = df.melt(id_vars=['file index', 'frame', 'GT'],
                      value_vars=map_vars,
                      var_name='Probability\nMap',
                      value_name='Metric')
    if not acc:
        long_df['Metric'] = (long_df['Metric'] / long_df['GT']) * 100
    else:
        long_df['Metric'] = ((long_df['GT'] - long_df['Metric']) / long_df['GT']) * 100
    long_df.rename(columns={'file index': 'File'}, inplace=True)

    fig, ax = plt.subplots(figsize=(6/2.54, 6/2.54))
    sns.boxplot(
        x='Probability\nMap',
        y='Metric',
        data=long_df,
        width=.5,
        ax=ax
    )
    sns.despine()
    if not acc:
        ax.set_ylabel(
            'Missed annotations [%]'
        )
    else:
        ax.set_ylabel(
            'Accuracy [%]'
        )

    print(long_df.groupby(['Probability\nMap'])['Metric'].describe())
    if method != 'both':
        print(f'Test for {map_vars[0]}')
        print(normaltest(df[map_vars[0]]))
        print(f'Test for {map_vars[1]}')
        print(normaltest(df[map_vars[1]]))
        print(mannwhitneyu(df[map_vars[0]].values, df[map_vars[1]].values))
        print(ttest_ind(df[map_vars[0]].values, df[map_vars[1]].values))


def plot_peak_find_distances(eval_file, method='edt', val_files=None):
    """
    Show the difference in distances between ground truth annotations and predicted annotations between running peak
    finding on raw probability maps or on processed maps.

    :param str eval_file: Path to .json file with peak finding evaluation data.
    :param str method: ['raw', 'eroded', 'edt'] Probability map processing method for which to plot mean annotation
        distances.
    :param list[str] val_files: List of filenames that are keys in the json data for which to show mean distances.
        If set to None, distances for all files will be plotted.
    """
    set_thesis_settings()

    with open(eval_file, 'r') as f:
        data = json.load(f)

    if val_files is None:
        val_files = list(data.keys())

    df_data = []
    for f_idx, filename in enumerate(list(data.keys())):
        for frame_idx, (dcom, dero, dedt, gt) in enumerate(
                zip(
                    data[filename]['dist_center'],
                    data[filename]['dist_eroded'],
                    data[filename]['dist_edt'],
                    data[filename]['num_gt'])):
            df_data.append({
                'file index': f_idx,
                'frame': frame_idx,
                'Raw': dcom,
                'Eroded': dero,
                'EDT': dedt,
                'GT': gt
            })

    if method == 'both':
        map_vars = ['Raw', 'Eroded', 'EDT']
    elif method == 'erosion':
        map_vars = ['Raw', 'Eroded']
    elif method == 'edt':
        map_vars = ['Raw', 'EDT']
    else:
        raise ValueError('Invalid value passed for `method`.')

    df = pd.DataFrame(df_data)
    long_df = df.melt(id_vars=['file index', 'frame', 'GT'],
                      value_vars=map_vars,
                      var_name='Probability\nMap',
                      value_name='Distance [px]')

    fig, ax = plt.subplots(figsize=(6/2.54, 6/2.54))
    sns.boxplot(
        x='Probability\nMap',
        y='Distance [px]',
        data=long_df,
        width=.5,
        ax=ax,
        # sym="",
    )
    sns.despine()

    print(long_df.groupby(['Probability\nMap'])['Distance [px]'].describe())


def plot_unet_training(hist_file):
    """
    Plot metrics recorded during training of U-Net/RU-Net.

    :param hist_file: Path to .json file of keras training history.
    """
    set_thesis_settings()

    with open(hist_file, 'r') as f:
        hist_data = eval(json.load(f))

    df = pd.DataFrame(hist_data)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Epoch'}, inplace=True)

    loss_df = df.melt(id_vars='Epoch',
                      value_vars=['loss', 'val_loss'],
                      var_name='T/V',
                      value_name='Loss')
    loss_df.replace(['loss', 'val_loss'], ['Training', 'Validation'], inplace=True)

    iou_df = df.melt(id_vars='Epoch',
                     value_vars=['iou_score', 'val_iou_score'],
                     var_name='T/V',
                     value_name='IoU')
    iou_df.replace(['iou_score', 'val_iou_score'], ['Training', 'Validation'], inplace=True)

    fig, axes = plt.subplots(2, 1, sharex=True)

    sns.lineplot(x='Epoch', y='IoU', hue='T/V', data=iou_df, ax=axes[0])
    axes[0].set_ylim([0.5, 1])
    axes[0].get_legend().remove()

    sns.lineplot(x='Epoch', y='Loss', hue='T/V', data=loss_df, ax=axes[1])
    axes[1].set_ylim(0, 0.5)
    sns.despine(fig=fig)

    plt.legend(loc='right', bbox_to_anchor=(1.25, 1.), frameon=False, labels=['Training', 'Validation'])


def plot_unets_kfold_training(kfold_dir, net='unet'):
    """
    Plot metrics recorded during k-fold training of U-Net and RU-Net.

    :param str kfold_dir: Path to directory of k-fold CV training. Must contain subdirectories
        that hold `unet*.json` and `runet*.json` files of training history.
    """
    set_thesis_settings()

    net_files = glob.glob(kfold_dir + f'**/{net}*.json')

    hists = []
    longest_epochs = -1
    for i in range(len(net_files)):
        with open(net_files[i], 'r') as f:
            hists.append(eval(json.load(f)))
            if len(hists[i]['loss']) > longest_epochs:
                longest_epochs = len(hists[i]['loss'])

    df_loss_cols = [f"{k} loss" for k in range(5)]
    df_loss_cols.extend([f"{k} val_loss" for k in range(5)])
    df_iou_cols = [f"{k} iou" for k in range(5)]
    df_iou_cols.extend([f"{k} val_iou" for k in range(5)])
    loss_df = pd.DataFrame(index=np.arange(longest_epochs), columns=df_loss_cols)
    iou_df = pd.DataFrame(index=np.arange(longest_epochs), columns=df_iou_cols)
    for k in range(5):
        loss_df[f"{k} loss"] = pd.Series(hists[k]['loss'])
        loss_df[f"{k} val_loss"] = pd.Series(hists[k]['val_loss'])
        iou_df[f"{k} iou"] = pd.Series(hists[k]['iou_score'])
        iou_df[f"{k} val_iou"] = pd.Series(hists[k]['val_iou_score'])
    loss_df.reset_index(inplace=True)
    iou_df.reset_index(inplace=True)

    long_loss = loss_df.melt(id_vars='index')
    long_iou = iou_df.melt(id_vars='index')
    long_loss[['k', 'dataset']] = long_loss['variable'].str.split(' ', expand=True)
    long_iou[['k', 'dataset']] = long_iou['variable'].str.split(' ', expand=True)

    fig, axes = plt.subplots(2, 1, sharex=True)

    sns.lineplot(x='index', y='value', hue='dataset', data=long_iou, ax=axes[0])
    axes[0].get_legend().remove()
    axes[0].set_ylabel('IoU Score')

    sns.lineplot(x='index', y='value', hue='dataset', data=long_loss, ax=axes[1])
    sns.despine(fig=fig)
    axes[1].set_ylabel('Dice Loss')
    axes[1].set_xlabel('Epoch')

    plt.legend(loc='right', bbox_to_anchor=(1.25, 1.), frameon=False, labels=['Training', 'Validation'])    

    mean_loss = long_loss.groupby(['index', 'dataset']).mean()
    mean_iou = long_iou.groupby(['index', 'dataset']).mean()

    print(mean_loss.groupby('dataset').describe())
    print(f"\nIndices of lowest loss:\n{mean_loss.groupby('dataset').idxmin()}\n")
    print(mean_iou.groupby('dataset').describe())
    print(f"\nIndices of highest iou:\n{mean_iou.groupby('dataset').idxmax()}\n")


def plot_unet_comp(df, param='iou', sides=False):
    """
    Plot comparison of IoU between U-Net and RU-Net.

    :param pd.DataFrame df: Evaluation dataframe.
    :param str param: ['iou', 'time'] Evaluation parameter to compare.
    :param bool sides: Split evaluations to compare results between left and right view sides.
    """
    set_thesis_settings()
    plt.rcParams['figure.figsize'] = (6/2.54, 6/2.54)

    val_vars = ['UNet IoU', 'RUNet IoU'] if param == 'iou' else ['UNet time', 'RUNet time']
    val_name = 'IoU' if param == 'iou' else 'Inference time [s]'

    long_df = df.melt(id_vars=['file idx', 'frame', 'side'],
                      value_vars=val_vars,
                      var_name='Network',
                      value_name=val_name)
    long_df['Network'].replace(val_vars, ['U-Net', 'RU-Net'], inplace=True)

    sns.boxplot(
        x='Network',
        y=val_name,
        data=long_df,
        width=.5,
        hue='side' if sides else None)
    sns.despine()
    plt.xlabel('')

    print(long_df.groupby('Network')[val_name].describe())


def plot_cluster_errors_comp(df, side_comp=False, violin=False):
    """
    Compare the amount of column clustering errors made by KMeans and GMM clustering methods.

    :param pd.DataFrame df: DataFrame containing evaluation data. Must be same format as created by
        `clustering_evaluation.ipynb`.
    :param bool side_comp: Show plot focusing on comparing the number of mistakes per side.
    """
    set_thesis_settings()

    figsize = (6/2.54, 6/2.54) if side_comp else (6/2.54, 6/2.54)
    plt.rcParams['figure.figsize'] = figsize

    miss_df = df.groupby(['file idx', 'frame', 'side'])[['KMeans miss', 'GMM miss']].sum().reset_index()
    miss_df.rename(columns={'side': 'Side'}, inplace=True)

    long_df = miss_df.melt(id_vars=['file idx', 'frame', 'Side'],
                           value_vars=['KMeans miss', 'GMM miss'],
                           var_name='Method',
                           value_name='Misses')
    long_df.replace(['KMeans miss', 'GMM miss'], ['K-means', 'GMM'], inplace=True)

    if not side_comp:
        if violin:
            long_df['all'] = ''
            sns.violinplot(x='all', y='Misses', data=long_df,
                           hue='Method', split=True, width=.75,
                           )
            plt.legend(bbox_to_anchor=(1.4, 1.), frameon=False)
        else:
            sns.boxplot(x='Method', y='Misses', data=long_df,
                        width=.5, flierprops=dict(marker='o'),
                        )
        plt.xlabel('')
    else:
        sns.boxplot(
            x='Side',
            y='Misses',
            data=long_df,
            hue='Method',
            width=.5,
        )
        plt.legend(bbox_to_anchor=(1.6, 1.), frameon=False)

    plt.ylabel('Errors / Grid')
    plt.ylim([0, 35])
    sns.despine()

    groupvar = 'Method' if not side_comp else ['Method', 'Side']
    print(long_df.groupby(groupvar)['Misses'].describe())

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def create_kmeans_example(save=False):
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from matplotlib import cm

    set_thesis_settings()
    plt.rcParams['figure.figsize'] = (7, 3.5)
    plt.rcParams['axes.grid'] = False
    plt

    # Create vertically spread but horizontally close data points of three classes
    n_samples = 100#1500
    random_state = 7

    X, y = make_blobs(
        n_samples=n_samples,
        centers=np.array([[-3, 0], [0, 0], [3, 0]]),
        cluster_std=0.5,
        random_state=random_state
    )
    X = X * [1, 4]

    # Create k-means and fit it to the data using standard euclidean distancing
    kmeans = KMeans(3, random_state=random_state)
    kmeans.fit(X, y)
    y_pred = kmeans.predict(X)
    # Create k-means and fit it to the data using weighted euclidean distancing
    kmeans2 = KMeans(3, random_state=random_state)
    kmeans2.fit(X*[1.5, 0.5])
    y_pred2 = kmeans2.predict(X)

    # Create map visualizing cluster results
    t = np.linspace(-10, 10)
    xx, yy = make_meshgrid(t, t)

    fig, axes = plt.subplots(1, 2)
    viridis = cm.get_cmap('viridis', 3)
    mrkrs = np.array(['X', 'o', '^'])
    plot_contours(axes[0], kmeans, xx, yy, alpha=0.4, cmap='viridis')
    for i in range(3):
        idcs = np.where(y == i)
        axes[0].scatter(X[idcs, 0], X[idcs, 1], c=viridis(y_pred[idcs]*1/2), cmap='viridis', marker=mrkrs[i], edgecolors='k', linewidths=0.5)
    axes[0].set_title('Euclidean distances')

    plot_contours(axes[1], kmeans2, xx, yy, alpha=0.4, cmap='viridis')
    for i in range(3):
        idcs = np.where(y == i)
        axes[1].scatter(X[idcs, 0], X[idcs, 1], c=viridis(y_pred2[idcs]*1/2), cmap='viridis', marker=mrkrs[i], edgecolors='k', linewidths=0.5)
    axes[1].set_title(r'(1.5, 0.5)-weighted Euclidean distances')

    sns.despine()


def plot_suture_grid_containment():
    """
    Plot distribution of width and height of manually labeled suture grid dimensions to demonstrate that expanding
    to 224x224px is sufficient to contain all suture grids.
    """
    set_thesis_settings()
    plt.rcParams['figure.figsize'] = (5, 2.5)

    with pd.HDFStore("endtoend_eval_df3.h5", mode="r") as store:
        idf = store.select("inference_df")

    idf = idf.drop_duplicates(['frame_id', 'side', 'file'])

    fig, ax = plt.subplots()
    sns.distplot(idf['gt_yolo_box_width'], bins=np.linspace(50, 250, 50), kde=False, color='#650021',
                 hist_kws=dict(alpha=.6), label='Width', ax=ax)
    sns.distplot(idf['gt_yolo_box_height'], bins=np.linspace(50, 250, 50), kde=False, color='#49759c',
                 hist_kws=dict(alpha=0.6), label='Height', ax=ax)

    r_ax = ax.twinx()
    r_ax.set_ylim([0, 1.1])

    heights, h_base = np.histogram(idf['gt_yolo_box_height'].values, bins=np.linspace(50, 250, 50))
    widths, w_base = np.histogram(idf['gt_yolo_box_width'].values, bins=np.linspace(50, 250, 50))
    heights_cum = np.cumsum(heights / np.sum(heights))
    widths_cum = np.cumsum(widths / np.sum(widths))
    r_ax.step(w_base[w_base < 224][1:], widths_cum[w_base[1:] < 224], c='#650021')
    r_ax.step(h_base[h_base < 224][1:], heights_cum[h_base[1:] < 224], c='#49759c')

    ax.axvline(x=224, ymin=0, ymax=1, c='k')
    ax.text(223, 64, "224px\nLimit", ha='right')
    ax.add_patch(Rectangle((224, 0), 100, 100, fill=False, edgecolor='r', hatch="///"))
    ax.legend(frameon=False)
    ax.set_xlabel('Suture grid width/height [px]')
    r_ax.set_xlabel('')
    ax.set_ylabel('Frequency')
    r_ax.set_ylabel('Cumulative frequency [%]')
    r_ax.set_yticklabels((r_ax.get_yticks() * 100).astype(int))
    ax.grid(False)
    r_ax.grid(False)
    sns.despine(top=True, right=False)


def plot_efficientnet_kfold_training(kfold_dir):
    """
    Plot metrics recorded during k-fold training of EfficientNetB0.

    :param str kfold_dir: Path to directory of k-fold CV training. Must contain subdirectories that hold
        the `history_top.json` and `history_tuned.json` files.
    """
    set_thesis_settings()

    top_files = glob.glob(kfold_dir + f"**/*top.json")
    tuned_files = glob.glob(kfold_dir + f"**/*tuned.json")

    top_hists = []
    tuned_hists = []
    for i in range(len(top_files)):
        with open(top_files[i], 'r') as f:
            top_hists.append(eval(json.load(f)))
        with open(tuned_files[i], 'r') as f:
            tuned_hists.append(eval(json.load(f)))

    df_columns = pd.MultiIndex.from_product([['train', 'validation'],
                                              ['loss', 'acc'],
                                              np.arange(len(top_files))])
    df_index = np.arange(50)
    top_df = pd.DataFrame(index=df_index, columns=df_columns)
    for i in range(len(top_files)):
        top_df['train', 'loss', i] = pd.Series(top_hists[i]['loss'])
        top_df['validation', 'loss', i] = pd.Series(top_hists[i]['val_loss'])
        top_df['train', 'acc', i] = pd.Series(top_hists[i]['accuracy'])
        top_df['validation', 'acc', i] = pd.Series(top_hists[i]['val_accuracy'])
    top_df.reset_index(inplace=True)

    tuned_df = pd.DataFrame(index=df_index, columns=df_columns)
    for i in range(len(tuned_files)):
        tuned_df['train', 'loss', i] = pd.Series(tuned_hists[i]['loss'])
        tuned_df['validation', 'loss', i] = pd.Series(tuned_hists[i]['val_loss'])
        tuned_df['train', 'acc', i] = pd.Series(tuned_hists[i]['accuracy'])
        tuned_df['validation', 'acc', i] = pd.Series(tuned_hists[i]['val_accuracy'])
    tuned_df.index = np.arange(50, 100)
    tuned_df.reset_index(inplace=True)

    toplong = top_df.melt(id_vars='index',
                          var_name=['dataset', 'metric', 'k'])
    tunedlong = tuned_df.melt(id_vars='index',
                              var_name=['dataset', 'metric', 'k'])

    lossdf = pd.concat([toplong[toplong['metric'] == 'loss'], tunedlong[tunedlong['metric'] == 'loss']])
    accdf = pd.concat([toplong[toplong['metric'] == 'acc'], tunedlong[tunedlong['metric'] == 'acc']])

    
    fig, axes = plt.subplots(2, 1, sharex=True)

    sns.lineplot(x='index', y='value', hue='dataset', data=accdf, ax=axes[0])
    axes[0].get_legend().remove()
    axes[0].set_ylabel('Accuracy [%]')

    sns.lineplot(x='index', y='value', hue='dataset', data=lossdf, ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')

    sns.despine()

    plt.legend(loc='right', bbox_to_anchor=(1.25, 1.), frameon=False, labels=['Training', 'Validation'])

    unfreeze_x = 49
    unfreeze_y = accdf[accdf['index'] == 49].iloc[0]['value']


def plot_efficientnet_kfold_acc(eval_file, crop=5, k_variance=False):
    """
    Plot accuracy of EfficientNetB0 suture sorting per patch.

    :param str eval_file: Path to HDFStore holding evaluation data.
    """
    set_thesis_settings()

    df = pd.read_hdf(eval_file)
    err_df = df.groupby(['k', 'file idx', 'frame', 'side']).sum().reset_index()
    small_df = err_df.copy()

    small_df['error'].clip(0, 1, inplace=True)
    err_df['error'].clip(0, crop, inplace=True)

    if k_variance:
        small_cnt = small_df.groupby('k')['error'].apply(
                lambda x: x.value_counts() * 100 / x.count()
            ).unstack(level=1).reset_index().melt(id_vars='k')
        cnt_df = err_df.groupby('k')['error'].apply(
                lambda x: x.value_counts() * 100 / x.count()
            ).unstack(level=1).reset_index().melt(id_vars='k')

        fig, axes = plt.subplots(1, 2, subplot_kw=dict(ylim=[0, 100]))
        sns.barplot(x='variable', y='value', data=small_cnt, ax=axes[0])
        for l, b in zip(axes[0].lines, axes[0].patches):
            x = l.get_xdata()[0]
            y = l.get_ydata()[1]
            y_box = b.get_bbox().get_points()[1, 1]
            axes[0].text(x, y + 5, "{:2.1f}%".format(y_box), horizontalalignment="center")

        sns.barplot(x='variable', y='value', data=cnt_df, ax=axes[1])
        for l, b in zip(axes[1].lines, axes[1].patches):
            x = l.get_xdata()[0]
            y = l.get_ydata()[1]
            y_box = b.get_bbox().get_points()[1, 1]
            axes[1].text(x, y + 5, "{:2.1f}%".format(y_box), horizontalalignment="center")

    else:
        fig, axes = plt.subplots(1, 2, gridspec_kw=dict(wspace=.5))
        percentage_countplot(small_df['error'], ax=axes[0])
        percentage_countplot(err_df['error'], ax=axes[1])

        axes[0].set_xlabel("")
        axes[0].set_xticks(ticks=np.arange(2))
        axes[0].set_xticklabels(labels=["No error", "Errors"])

        axes[1].set_xlabel('# Sorting errors')
        axes[1].set_xticks(ticks=np.arange(crop + 1))
        axes[1].set_xticklabels(labels=list(range(crop)) + [f'$\geq${crop}'])


def create_suture_grid_showcase():
    set_thesis_settings()

    np.random.seed(1337)

    rois_files = glob.glob('daten/fuer_nn/*.rois')
    identifiers = [f.split('Cam')[0] for f in rois_files]

    # Get the first index of each unique hemi-larynx identifier file
    seen = set()
    unique_idcs = []
    for i, f in enumerate(identifiers):
        if f not in seen:
            unique_idcs.append(i)
            seen.add(f)

    showcase_files = np.asarray(rois_files)[unique_idcs].tolist()

    fig, axes = plt.subplots(2, 6, figsize=(7, 3), gridspec_kw=dict(hspace=-.15))
    for i in range(11):
        row, col = np.unravel_index(i, (2, 6))
        
        rois_file = showcase_files[i]
        file_parts = os.path.basename(rois_file).split('_')
        model = file_parts[1] if 'Human' in file_parts else file_parts[0]
        patches = utils.extract_rois_patch(rois_file, 0, (224, 224), True)
        patch_idx = np.random.randint(0, patches.shape[0])
        show_patch = patches[patch_idx]
        
        ax = axes[row][col]
        ax.imshow(show_patch, cmap='gray')
        ax.axis('off')
        ax.set_title(model)
    axes[1, 5].axis('off')  # Hide the last empty plot
    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.005, right=0.995, hspace=0.2, wspace=0.2)


def plot_normalization_example():
    set_thesis_settings()
    plt.rcParams['lines.markersize'] = 3.0

    rois = utils.load_rois_patch_relative('daten/fuer_nn/Human_I180919_Cam_16904_Cine3_100frames.rois', (224, 224))
    raw_rois = np.ma.masked_equal(rois[2, :35] * [224, 224], -224)
    norm_rois = utils.normalize_peaks(raw_rois)
    pca_rois = utils.bbox_pca_transform_peaks(norm_rois.compressed().reshape(-1, 2))

    fig, axes = plt.subplots(1, 3, figsize=(18/2.54, 5/2.54))

    axes[0].scatter(raw_rois[:, 0], raw_rois[:, 1])
    axes[0].set_xlim([0, 224])
    axes[0].set_ylim([224, 0])
    axes[0].set_aspect(1)
    axes[0].set_title("Unprocessed")

    axes[1].scatter(norm_rois[:, 0], norm_rois[:, 1])
    axes[1].set_xlim([-3, 3])
    axes[1].set_ylim([3, -3])
    axes[1].set_aspect(1)
    axes[1].set_title("Normalized")

    axes[2].scatter(pca_rois[:, 0], pca_rois[:, 1])
    axes[2].set_xlim([-3, 3])
    axes[2].set_ylim([3, -3])
    axes[2].set_aspect(1)
    axes[2].set_title("PCA Transformed")


def plot_cluster_sort_errors(df):
    """
    Plot comparison of grid sort after clustering between K-means and GMM clustering.

    :param df: DataFrame created by `clustering_evaluation` notebook holding evaluated sorting observations.
    """
    set_thesis_settings()

    error_df = df.melt(id_vars=['file idx', 'frame', 'side'],
                       value_vars=['KMeans errors', 'GMM errors'],
                       var_name='Clustering method',
                       value_name='Errors')
    error_df.replace(['KMeans errors', 'GMM errors'], ['K-means', 'GMM'],
                     inplace=True)

    plt.figure(figsize=(6/2.54, 6/2.54))
    sns.boxplot(
        x='Clustering method',
        y='Errors',
        data=error_df,
        width=.5
    )
    sns.despine()

    print(error_df.groupby('Clustering method')['Errors'].describe())


def plot_cluster_sort_correlation(df):
    """
    Plot correlation between amount of errors during column clustering with amount of errors for final grid sorting.

    :param df: DataFrame created by `clustering_evaluation` notebook holding evaluated sorting observations.
    """
    set_thesis_settings()

    s1 = df.melt(id_vars=['file idx', 'frame', 'side'],
                 value_vars=['KMeans col errors', 'GMM col errors'],
                 var_name='Method',
                 value_name='Col errors')

    s2 = df.melt(id_vars=['file idx', 'frame', 'side'],
                 value_vars=['KMeans errors', 'GMM errors'],
                 var_name='Method',
                 value_name='Errors')
    s2.drop(columns='Method', inplace=True)

    corr_df = s1.merge(s2, on=['file idx', 'frame', 'side'])
    corr_df.replace(['KMeans col errors', 'GMM col errors'], ['K-means', 'GMM'],
                    inplace=True)

    plt.subplots(figsize=(10/2.54, 6/2.54))
    sns.pointplot(
        x='Col errors',
        y='Errors',
        hue='Method',
        data=corr_df,
        dodge=0.4,
        scale=0.6,
        order=np.arange(19),
        errwidth=1.2,
    )
    sns.despine()
    plt.ylim([0, 35])
    plt.legend(frameon=False, bbox_to_anchor=(1, 1))
    plt.xlabel("Clustering errors")
    plt.ylabel("Assignment errors")


def plot_errors_all_sort_methods(clustering_hdf, efficientnet_hdf):
    set_thesis_settings()

    clus_df = pd.read_hdf(clustering_hdf)
    eff_df = pd.read_hdf(efficientnet_hdf)
    efferr_df = eff_df.groupby(['file', 'frame', 'side']).sum().reset_index()

    # Insert the EfficientNetB0 errors into the other DataFrame
    clus_df['EfficientNetB0'] = efferr_df['error']

    # Create long-form data from DataFrame
    long_df = clus_df.melt(id_vars=['file', 'frame', 'side'],
                           value_vars=['KMeans errors', 'GMM errors', 'EfficientNetB0'],
                           var_name='Method',
                           value_name='Assignment errors')
    long_df.replace(['KMeans errors', 'GMM errors'],
                    ['Clustering sort\n(K-means)', 'Clustering sort\n(GMM)'],
                    inplace=True)

    plt.figure(figsize=(9/2.54, 6/2.54))
    sns.boxplot(x='Method', y='Assignment errors', data=long_df, width=.5)
    sns.despine()
    plt.xlabel("")