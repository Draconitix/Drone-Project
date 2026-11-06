from pysimverse import Drone
import time

#initialization
drone = Drone()
drone.connect()
drone.take_off()

drone.rotate(48)
drone.move_forward(360)

drone.land()
time.sleep(1)