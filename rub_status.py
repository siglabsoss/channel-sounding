from util import *
import time

if __name__ == '__main__':
    rub = initSerial('/dev/ttyUSB1', 9600)
    print rub

    for _ in range(2):
        rub.write('\r\n')
        time.sleep(0.1)

    jout = readFor(rub, 0.5)
    print "first", jout
    asciiPrintString(jout)

    rub.write('id?\r\n')

    jout = readFor(rub, 0.5)


    rub.write('st?\r\n')

    stout = readFor(rub, 0.5)

    print stout

    rub.close()


