import unittest
from sp.data.sp_data import SPData


class MyTestCase(unittest.TestCase):
    def test__intersect_overlapping_wall(self):

        line_begin = (0, 0, 0, 0)
        line_end = (1, 0, 0, 0)

        wall_begin = (0.25, 0)
        wall_end = (0.75, 0)

        line = (line_begin, line_end)
        wall = (wall_begin, wall_end, 1)
        # Could be also False, not clear if wall along line is possible or not
        self.assertEqual(SPData._intersect(line, wall), True)  # add assertion here

    def test__intersect_start_points_far_apart(self):
        line_begin = (0, 0, 0, 0)
        line_end = (1, 10, 0, 0)

        wall_begin = (0.5, 10)
        wall_end = (0.5, -1)

        line = (line_begin, line_end)
        wall = (wall_begin, wall_end, 1)

        self.assertEqual(SPData._intersect(line, wall), True)  # add assertion here

    def test__intersect_negative_slope_y_direction(self):
        line_begin = (0, 0, 2, 0)
        line_end = (0, 2, 0, 0)

        wall_begin = (1, 0.5)
        wall_end = (-1, 0.5)

        line = (line_begin, line_end)
        wall = (wall_begin, wall_end, 1.5)

        self.assertEqual(SPData._intersect(line, wall), False)  # add assertion here

    def test__intersect_negative_slope_x_direction(self):
        line_begin = (0, 0, 2, 0)
        line_end = (2, 0, 0, 0)

        wall_begin = (0.5, 1)
        wall_end = (0.5, -1)

        line = (line_begin, line_end)
        wall = (wall_begin, wall_end, 1.5)

        self.assertEqual(SPData._intersect(line, wall), False)  # add assertion here

if __name__ == '__main__':
    unittest.main()
