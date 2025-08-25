from unittest import TestCase
from unittest import mock
import pandas as pd
import plotly.express

from lb.plotting.lb_plot import LBPlot
from unittest.mock import ANY, patch

class TestLBPlot(TestCase):
    example_records = [
        dict(Truck=0, Capacity=3, TotalLoad=60, NumberOfUnitsProduct_0=1,
             NumberOfUnitsProduct_1=1, NumberOfUnitsProduct_2=1,
             LoadOfProduct_0=10, LoadOfProduct_1=20, LoadOfProduct_2=30),

        dict(Truck=1, Capacity=3, TotalLoad=50, NumberOfUnitsProduct_0=1,
             NumberOfUnitsProduct_1=2, NumberOfUnitsProduct_2=0,
             LoadOfProduct_0=10, LoadOfProduct_1=40, LoadOfProduct_2=0),

        dict(Truck=2, Capacity=3, TotalLoad=60, NumberOfUnitsProduct_0=0,
             NumberOfUnitsProduct_1=3, NumberOfUnitsProduct_2=0,
             LoadOfProduct_0=0, LoadOfProduct_1=60, LoadOfProduct_2=0)
    ]
    example_data_frame = pd.DataFrame.from_records(example_records)
    mock_data_object = mock.Mock()
    mock_data_object.product_types = range(3)

    mock_evaluation = mock.Mock()
    mock_evaluation.interpret_solution.return_value = example_data_frame

    @mock.patch("plotly.express.timeline")
    def test_plot_solution_call(self, mock_function):
        reference_records = [
            dict(Truck=0, Start=0, Finish=30, ProductType=2, delta=30),
            dict(Truck=0, Start=30, Finish=50, ProductType=1, delta=20),
            dict(Truck=0, Start=50, Finish=60, ProductType=0, delta=10),

            dict(Truck=1, Start=0, Finish=20, ProductType=1, delta=20),
            dict(Truck=1, Start=20, Finish=40, ProductType=1, delta=20),
            dict(Truck=1, Start=40, Finish=50, ProductType=0, delta=10),

            dict(Truck=2, Start=0, Finish=20, ProductType=1, delta=20),
            dict(Truck=2, Start=20, Finish=40, ProductType=1, delta=20),
            dict(Truck=2, Start=40, Finish=60, ProductType=1, delta=20),

        ]

        reference_data_frame = pd.DataFrame.from_records(reference_records)
        LBPlot(self.mock_data_object, self.mock_evaluation).plot_solution()

        df_called_with = mock_function.call_args_list[0][0][0]

        mock_function.assert_called_with(ANY,
                                         x_start="Start", x_end="Finish", y="Truck",
                                         text="ProductType", color="ProductType")
        self.assertTrue(reference_data_frame.equals(df_called_with))

    @mock.patch("plotly.express.timeline")
    def test_average_bar(self, mock_timeline):
        mock_fig = mock_timeline.return_value
        mock_fig.add_vline = mock.Mock()

        LBPlot(self.mock_data_object, self.mock_evaluation).plot_solution()
        kwargs = mock_fig.add_vline.call_args.kwargs
        position = kwargs["x"]
        self.assertAlmostEqual(170/3, position)