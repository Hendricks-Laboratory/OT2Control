import time

from spectro_interface import PlateReader

pr = PlateReader()
pr.plateout()
time.sleep(4)
pr.platein()
