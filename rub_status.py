from util import *
import time


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

if __name__ == '__main__':
    jak = initSerial('/dev/ttyUSB1', 9600)
    print jak

    for _ in range(2):
        jak.write('\n')
        time.sleep(0.3)

    jout = readFor(jak, 1)
    print "first", jout
    asciiPrintString(jout)

    jak.write('i')
    time.sleep(0.5)
    jak.write('d')
    time.sleep(0.5)
    jak.write('?')
    time.sleep(0.5)
    jak.write('\n')
    time.sleep(0.5)
    jak.write('\n')

    jout = readFor(jak, 2)
    print jout
    asciiPrintString(jout)


    jak.close()

