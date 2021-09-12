import tfhelper.core as core
import matplotlib.pyplot as plt

plt.ioff()


def test_MySequence():
    ms1, ms2 = core.create_seqs(core.EXAMPLE_DIR, 2)
    for i in range(len(ms1)):
        ms1.visualize(i)
        plt.show()
    for i in range(len(ms2)):
        ms2.visualize(i)
        plt.show()
