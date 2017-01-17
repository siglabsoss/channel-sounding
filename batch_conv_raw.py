from sigmath import *
import scipy.io as sio
import argparse
import sys
import os
from subprocess import call

def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def conv_file(iin, oout):

    # splits off the filename, and then removes the extension
    varname = 'data_' + os.path.splitext(os.path.split(pi)[1])[0]

    fi = open(iin, 'r')

    fsz = getSize(iin)

    print "file", iin, "has", fsz, "bytes"

    print "matlab variable:", varname

    bytes = fi.read(fsz)
    lenread = len(bytes)

    print "bytes in ram", fsz, lenread

    fi.close()

    complex = raw_to_complex_multi(bytes)

    print 'data in complex'

    bytes = None

    print 'bytes dumped from ram'

    sio.savemat(oout, {varname: complex}, format='4', do_compression=True)

    print 'bytes saved to disk'







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
    # ansy = 'y'

    if ansy != 'y':
        print "quitting"
        sys.exit(1)

    call(['mkdir', '-p', path_out])

    # conv_file('/home/ubuntu/channel-sounding/trunc.raw', '/home/ubuntu/out.mat')


    full_paths = []
    full_paths_out = []

    for root, dirs, files in os.walk(path_in):
        for f in files:
            if f[-4:] == '.raw':
                full_paths.append(os.path.join(root, f))


        # print '--------'
        # print root, "-", os.path.join(path_out, os.path.basename(root))
        if( os.path.abspath(root) != os.path.abspath(path_in)):
            compfolder = os.path.join(path_out, os.path.basename(root)) # same foldername in output dir
            call(['mkdir', '-p', compfolder])
            print 'made', compfolder

            # print os.path.join(compfolder, files[0])

            for f in files:
                if f[-4:] == '.raw':
                    full_paths_out.append(os.path.join(compfolder, f))

            # print "pass"
        # data2 = [os.path.join(root, path_out) for f in files]
        # print data2
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

    assert(len(full_paths) == len(full_paths_out))

    for i in range(len(full_paths)):
        pi = full_paths[i]
        po = full_paths_out[i]
        # print pi

        conv_file(pi,po)

        # print os.path.splitext(os.path.split(pi)[1])[0]

        # this prints the common characters on the rhs
        # if this isn't correct we are saving the wrong input to output files as a consequence of how full_paths vs full_paths_out was made
        # trick = os.path.commonprefix([pi[::-1],po[::-1]])
        # print trick[::-1]



    # print repr(full_paths)

    # for path in full_paths:




    # sio.savemat('testfile.mat', {'data': data})



