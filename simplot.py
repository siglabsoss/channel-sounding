from sigmath import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Plot")
    parser.add_argument('file', nargs='+', help='Path to a raw file')
    args = parser.parse_args()
    path = vars(args)['file'][1]

    f = open(path)
    # bytes = f.read(9999999999)
    bytes = f.read(999999)
    print "read", len(bytes)

    data = []

    llen = len(bytes)
    # llen = 100

    data = [None] * (llen/8)

    print "data is", len(data)

    print "allocated empty"

    for i in range(0,llen-16,8):
        # data.append(raw_to_complex(bytes[i:i+8]))
        data[i/8] = np.real(raw_to_complex(bytes[i:i+8]))

        # print (i / float(llen)*1000)

        # if( round(i / llen * 1000) )

    print "finished converting"

    # nplotdots(np.real(data[0:100]))
    nplot(np.real(data))
    nplotshow()
