#!/usr/bin/python

import sys
import time
import serial
import array

# debug print function that prints byte array
def asciiPrintBytes(ba):
    for b in ba:
        print b
    print ''

def asciiPrintString(s):
    for c in s:
        print ord(c)



def initSerial(pt, baud):
    # configure the serial connections (the parameters differs on the device you are connecting to)
    ser = serial.Serial(
        port=pt,
        baudrate=baud,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )


    # why is this needed? are we not closing properly?
    if ser.isOpen():
        ser.close()


    ser.open()

    return ser
