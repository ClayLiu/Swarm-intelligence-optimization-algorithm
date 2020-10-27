import numpy as np
import matplotlib.pyplot as plt

from Utils.test_functions import *

for tester in unimodal_function:
    tester().plot(hold_on = True)

# for tester in multimodal_functions:
#     tester().plot(hold_on = True)

plt.show()