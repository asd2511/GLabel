import unittest
import random

from PyQt5.QtCore import QPointF

from nn import postprocessing


class PostProcessingTest(unittest.TestCase):

    def test_fixup_sortings_ordered_col(self):
        """Correctly ordered columns should not be changed"""
        # Stereo grid with one column per side, single frame
        rois = [[QPointF(0, 3), QPointF(0, 2), QPointF(0, 1),
                 QPointF(1, 3), QPointF(1, 2), QPointF(1, 1)]]
        proc_rois = postprocessing.fixup_sortings(rois, 1, 3, stereo=True, fix_rows=False)
        self.assertListEqual(rois, proc_rois)

        # Mono grid with one column, single frame
        rois = [[QPointF(0, 3), QPointF(0, 2), QPointF(0, 1)]]
        proc_rois = postprocessing.fixup_sortings(rois, 1, 3, stereo=False, fix_rows=False)
        self.assertListEqual(rois, proc_rois)

    def test_fixup_sortings_ordered_row(self):
        """Correctly ordered rows should not be changed"""
        # Stereo grid with one row per side, single frame
        rois = [[QPointF(0, 1), QPointF(1, 3), QPointF(2, 1),
                QPointF(10, 4), QPointF(11, 1), QPointF(12, 5)]]
        proc_rois = postprocessing.fixup_sortings(rois, 3, 1, stereo=True, fix_cols=False)
        self.assertListEqual(rois, proc_rois)

        # Mono grid with one row, single frame
        rois = [[QPointF(0, 1), QPointF(1, 3), QPointF(2, 1)]]
        proc_rois = postprocessing.fixup_sortings(rois, 3, 1, stereo=False, fix_cols=False)
        self.assertListEqual(rois, proc_rois)

    def test_fixup_sortings_vert_error(self):
        """Sortings with vertically incorrect orders should be changed to fit vertical order (fixing columns)"""
        # Stereo grid with one column per side, single frame
        rois = [[QPointF(0, 3), QPointF(0, 1), QPointF(0, 2),
                 QPointF(1, 2), QPointF(1, 1), QPointF(1, 3)]]
        corr_rois = [[QPointF(0, 3), QPointF(0, 2), QPointF(0, 1),
                      QPointF(1, 3), QPointF(1, 2), QPointF(1, 1)]]
        proc_rois = postprocessing.fixup_sortings(rois, 1, 3, stereo=True, fix_rows=False)
        self.assertListEqual(corr_rois, proc_rois)

        # Mono grid with one column, single frame
        rois = [[QPointF(1, 2), QPointF(1, 1), QPointF(1, 3)]]
        corr_rois = [[QPointF(1, 3), QPointF(1, 2), QPointF(1, 1)]]
        proc_rois = postprocessing.fixup_sortings(rois, 1, 3, stereo=False, fix_rows=False)
        self.assertListEqual(corr_rois, proc_rois)

    def test_fixup_sortins_hor_error(self):
        """Sortings with horizontally incorrect orders should be changed to fit horizontal order (fixing rows)"""
        # Stereo grid with one row per side, single frame
        rois = [[QPointF(1, 1), QPointF(0, 3), QPointF(2, 1),
                 QPointF(12, 4), QPointF(11, 1), QPointF(10, 5)]]
        corr_rois = [[QPointF(0, 3), QPointF(1, 1), QPointF(2, 1),
                      QPointF(10, 5), QPointF(11, 1), QPointF(12, 4)]]
        proc_rois = postprocessing.fixup_sortings(rois, 3, 1, stereo=True, fix_cols=False)
        self.assertListEqual(corr_rois, proc_rois)

        # Mono grid with one row, single frame
        rois = [[QPointF(1, 3), QPointF(2, 1), QPointF(0, 1)]]
        corr_rois = [[QPointF(0, 1), QPointF(1, 3), QPointF(2, 1)]]
        proc_rois = postprocessing.fixup_sortings(rois, 3, 1, stereo=False, fix_cols=False)
        self.assertListEqual(corr_rois, proc_rois)

    def test_fixup_sortings_vert_random(self):
        """Radnomly generated coordinates for ROIs should become vertically sorted"""
        # 5 columns, 5 rows, 10 frames
        rows = 5
        cols = 5
        frames = 10

        # Stereo grid
        total_placements = ((rows * cols) * 2) * frames
        xs = random.sample(range(768), total_placements)
        ys = random.sample(range(768), total_placements)
        rand_points = [QPointF(x, y) for (x, y) in zip(xs, ys)]
        rois = [[p for p in rand_points[i*rows*cols*2:(i+1)*rows*cols*2]] for i in range(frames)]
        proc_rois = postprocessing.fixup_sortings(rois, 5, 5, True, fix_rows=False)

        for frame in range(frames):
            for col in range(cols):
                self.assertListEqual(proc_rois[frame][col::cols][:rows],
                                     sorted(rois[frame][col::cols][:rows], key=lambda p: p.y(), reverse=True),
                                     f"Difference for frame {frame}, column {col}")
                self.assertListEqual(proc_rois[frame][col::cols][rows:],
                                     sorted(rois[frame][col::cols][rows:], key=lambda p: p.y(), reverse=True),
                                     f"Difference for frame {frame}, column {col} (right side)")

        # Mono grid
        total_placements = rows * cols * frames
        xs = random.sample(range(768), total_placements)
        ys = random.sample(range(768), total_placements)
        rand_points = [QPointF(x, y) for (x, y) in zip(xs, ys)]
        rois = [[p for p in rand_points[i*rows*cols:(i+1)*rows*cols]] for i in range(frames)]
        proc_rois = postprocessing.fixup_sortings(rois, 5, 5, False, fix_rows=False)

        for frame in range(frames):
            for col in range(cols):
                self.assertListEqual(proc_rois[frame][col::cols],
                                     sorted(rois[frame][col::cols], key=lambda p: p.y(), reverse=True),
                                     f"Difference for frame {frame}, column {col}")

    def test_fixup_sortings_hor_random(self):
        # 5 rows, 5 columns, 10 frames
        rows = 5
        cols = 5
        frames = 10

        # Stereo grid
        total_placements = ((rows*cols) * 2) * frames
        xs = random.sample(range(768), total_placements)
        ys = random.sample(range(768), total_placements)
        rand_points = [QPointF(x, y) for (x, y) in zip(xs, ys)]
        rois = [[p for p in rand_points[i*rows*cols*2:(i+1)*rows*cols*2]] for i in range(frames)]
        proc_rois = postprocessing.fixup_sortings(rois, 5, 5, True, fix_cols=False)

        for frame in range(frames):
            for row in range(rows*2):
                self.assertListEqual(proc_rois[frame][row * rows:(row + 1) * rows],
                                     sorted(rois[frame][row * rows:(row + 1) * rows], key=lambda p: p.x()),
                                     f"Difference for frame {frame}, column {row} (right side)")

        # Mono grid
        total_placements = rows * cols * frames
        xs = random.sample(range(768), total_placements)
        ys = random.sample(range(768), total_placements)
        rand_points = [QPointF(x, y) for (x, y) in zip(xs, ys)]
        rois = [[p for p in rand_points[i * rows * cols:(i + 1) * rows * cols]] for i in range(frames)]
        proc_rois = postprocessing.fixup_sortings(rois, 5, 5, False, fix_cols=False)

        for frame in range(frames):
            for row in range(rows):
                self.assertListEqual(proc_rois[frame][row * rows:(row+1) * rows],
                                     sorted(rois[frame][row * rows:(row + 1) * rows], key=lambda p: p.x()),
                                     f"Difference for frame {frame}, column {row} (right side)")


if __name__ == '__main__':
    unittest.main()