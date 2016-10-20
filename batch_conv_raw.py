from sigmath import *
import scipy.io as sio
import argparse
import sys
import os


def conv_file(iin,oout):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".raw -> .mat Data Converter")
    parser.add_argument('in', nargs='+', help='infile path')
    parser.add_argument('out', nargs='+', help='outfile path')

    args = parser.parse_args()

    # print vars(args)

    path_in = vars(args)['in'][0]
    path_out = vars(args)['out'][0]

    # print path_out

    ansy = raw_input('Converting all files under:\n\t' + path_in + "\n to \n\t" + path_out + "\n\npress y to continue: ")

    if ansy != 'y':
        print "quitting"
        sys.exit(1)


    full_paths = []

    for root, dirs, files in os.walk(path_in):
        for f in files:
            full_paths.append(os.path.join(root, f))


        # print '--------'
        # print os.path.basename(root)
        # print root
        # print files
        # data = [os.path.join(root, f) for f in files]
        # print data
        # print ''
        # print ''
        # print ''

        # if os.path.basename(root) != 'modules':
        #     continue

    print repr(full_paths)


    data = [1-1j,1+0.1j,3j]

    # sio.savemat('testfile.mat', {'data': data})



