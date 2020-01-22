import epics
import time

camera_prefix = '7bm_pg4'
rot_motor = epics.Motor('7bmb1:aero:m3')

epics.caput('7bma1:rShtrA:Open',1)

time.sleep(3.0)
epics.caput(camera_prefix + ':cam1:Acquire',1)
time.sleep(0.3)
epics.move(180, relative=True)
epics.caput(camera_prefix + ':cam1:Acquire',1)
time.sleep(0.5)
epics.caput('7bma1:rShtrA:Close',1)
epics.move(-180, relative=True)

