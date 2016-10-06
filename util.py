#!/usr/bin/python

import sys
import time
import serial
import array
from subprocess import call

# debug print function that prints byte array
def asciiPrintBytes(ba):
    for b in ba:
        print b
    print ''

def asciiPrintString(s):
    for c in s:
        print ord(c)


def readFor(ser, seconds):

    out = ''
    st = time.time()
    ct = 0
    while (time.time() - st) < seconds:
        #print st
        #time.sleep(0.1)
        wt = ser.inWaiting()
        if wt > 0:
            out += ser.read(wt)
            ct += wt

    print "read", ct, "chars"
    return out


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

    # this is annoying but this required to get python to use the same settings that screen does
    # without this all python can read back is 0xff bytes
    # this was generated by comparing the outputs of stty -f /dev/ttyUSB0 after screen and python touched the port
    call(['stty', '-F', pt, 'echoe', 'echok', 'echoctl', 'echoke', 'iexten'])

    return ser
