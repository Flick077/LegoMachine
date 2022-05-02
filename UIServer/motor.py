import RPi.GPIO as GPIO
from time import sleep
from gpiozero import Servo

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Blink LED
GPIO.setup(14, GPIO.OUT, initial=GPIO.LOW)
# Shaker table motor
GPIO.setup(4, GPIO.OUT, initial=GPIO.LOW)
# Conveyor belt motor
GPIO.setup(15, GPIO.OUT, initial=GPIO.LOW)
# Servo
#servo = Servo(21)
    
def ConveyorBeltOn() :
    GPIO.output(15, GPIO.HIGH)
    
def ConveyorBeltOff() :
    GPIO.output(15, GPIO.LOW)
    
def ShakerTableOn() :
    GPIO.output(4, GPIO.HIGH)
    
def ShakerTableOff() :
    GPIO.output(4, GPIO.LOW)
    
def HopperDoorOpen() :
    #servo.value = 0
    pass
    
def HopperDoorClose() :
    #servo.value = 1
    pass

if __name__ == "__main__":
    while True:
        GPIO.output(14, GPIO.HIGH) # LED
        ConveyorBeltOn()
        ShakerTableOn()
        HopperDoorOpen()
        sleep(2.5)
        
        GPIO.output(14, GPIO.LOW)
        ConveyorBeltOff()
        ShakerTableOff()
        HopperDoorClose()
        sleep(2.5)

    