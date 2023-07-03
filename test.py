import unittest
from intersection_result import IntersectionResult
import matplotlib.pyplot as plt


class TestMyClass(unittest.TestCase):
    def test_plot_chart(self):
        my_class = IntersectionResult(
            [ 0.23757148, -0.31370679, -0.03595497],
            [ 0.17105426, -0.44368112, -0.06540859],
            [[0.10470458, -0.71442705, -1.],
             [0.10470458, -0.30123359, -1.],
             [0.33731371, -0.30123359,  1.],
            [0.33731371, -0.71442705, 1.]],
            [[0.12983863055706024, -0.5709596276283264, -0.2713238298892975],
             [0.13512705266475677, -0.6069574952125549, -0.2592221796512604],
             [0.1344548612833023, -0.6067160964012146, -0.2616119682788849],
             [0.13515008985996246, -0.6069170832633972, -0.25922343134880066],
             [0.10576529800891876, -0.6069518327713013, -0.27731627225875854],
             [0.10773640871047974, -0.6076177954673767, -0.2772231101989746],
             [0.10709403455257416, -0.6067482829093933, -0.2731925845146179],
             [0.13086019456386566, -0.5973066687583923, -0.15236906707286835],
             [0.009617284871637821, -0.5919349789619446, -0.2143862247467041],
             [0.12181683629751205, -0.5463674664497375, -0.2369326502084732],
             [0.08709125965833664, -0.5434131622314453, -0.2542240023612976],
             [0.17105425894260406, -0.44368112087249756, -0.06540858745574951],
             [-0.09337188303470612, -0.4381471276283264, -0.19988207519054413],
             [0.23757147789001465, -0.31370678544044495, -0.03595496714115143],
             [-0.05994034558534622, -0.24914821982383728, -0.20961353182792664],
             [0.24782538414001465, -0.25122079253196716, -0.12278608232736588],
             [0.15603584051132202, -0.18807455897331238, -0.25777754187583923],
             [0.24046100676059723, -0.23465462028980255, -0.14799445867538452],
             [0.20126596093177795, -0.16512896120548248, -0.2913084626197815],
             [0.23256592452526093, -0.2546633183956146, -0.16779805719852448],
             [0.2267702966928482, -0.2047601342201233, -0.2844429910182953],
             [0.23655296862125397, -0.2531210482120514, -0.13774491846561432],
             [0.17975959181785583, -0.1917625069618225, -0.2581232786178589],
             [0.09901656210422516, 0.0002360180951654911, 0.0327768437564373],
             [-0.09586663544178009, -0.00809959415346384, -0.030353082343935966],
             [0.1475282907485962, -0.0700402781367302, -0.09338225424289703],
             [-0.07977931201457977, -0.03436443954706192, -0.1941649615764618],
             [0.11019295454025269, 0.21591390669345856, 0.1514236181974411],
             [-0.08875568211078644, 0.39111045002937317, 0.01229464914649725],
             [0.10915204882621765, 0.23710040748119354, 0.17398570477962494],
             [-0.0812041312456131, 0.4093189835548401, 0.04113420099020004],
             [0.19245269894599915, 0.19445422291755676, -0.021904749795794487],
             [-0.04760047793388367, 0.41672319173812866, -0.15323373675346375]],
            None,
            [ 0.21570111, -0.35644138, -0.0456391 ],
            [ 0.01025391,  0.06248599, -0.08683112],
            [0.06651722, 0.12997434, 0.02945362],
        )
        fig = my_class.create_fig(plt.figure(figsize=[3, 3]))
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_figheight(), 3.0)
        self.assertEqual(fig.get_figwidth(), 3.0)


if __name__ == "__main__":
    unittest.main()