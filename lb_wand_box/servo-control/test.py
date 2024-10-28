import sys

args = sys.argv

import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BOARD)

GPIO.setup(11, GPIO.OUT)
p = GPIO.PWM(11, 50)
p.start(0)

if args[1]:
    p.ChangeDutyCycle(float(args[1]));
    print(args[1]);
    sleep(.2);

# p.ChangeFrequence(1);
p.stop()
GPIO.cleanup()

# purple =      9 (ground?)
# red =         2 (5v)
# yellow =      11 (io)