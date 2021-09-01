import tfhelper.core as core
import matplotlib.pyplot as plt

plt.ioff()


def test_MySequence():
    ms = core.MySequence(core.EXAMPLE_DIR, 4)
    for i in range(3):
        ms.visualize(i)
        plt.show()
