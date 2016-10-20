import struct
import numpy as np
import collections
import string
import socket
import errno
import sys
import logging
import csv
import random
import time
import math
from numpy import real, imag, arange
from numpy.fft import fft, ifft, fftshift
from numpy.fft import fft, ifft, fftshift
from numpy.random import randint
import itertools
from itertools import chain, repeat, islice, izip
import operator
from scipy import ndimage
import matplotlib.pyplot as plt
import difflib
from skimage import io, filters
import Image
import hashlib
import zmq
import scipy


# converts string types to complex
def raw_to_complex(str):
    f1 = struct.unpack('%df' % 1, str[0:4])
    f2 = struct.unpack('%df' % 1, str[4:8])

    f1 = f1[0]
    f2 = f2[0]
    return f1 + f2*1j

def complex_to_raw(n):

    s1 = struct.pack('%df' % 1, np.real(n))
    s2 = struct.pack('%df' % 1, np.imag(n))

    return s1 + s2

# converts complex number to a pair of int16's, called ishort in gnuradio
def complex_to_ishort(c):
    short = 2**15-1
    re = struct.pack("h", np.real(c)*short)
    im = struct.pack("h", np.imag(c)*short)
    return re+im

def complex_to_ishort_multi(floats):
    rr = np.real(floats)
    ii = np.imag(floats)
    zz = np.array((rr,ii)).transpose()
    zzz = zz.reshape(len(floats)*2) * (2**15-1)
    bytes = struct.pack("%sh" % len(zzz), *zzz)
    return bytes

def ishort_to_complex_multi(ishort_bytes):
    packed = struct.unpack("%dh" % int(len(ishort_bytes)/2), ishort_bytes)
    rere = sig_everyn(packed, 2, 0)
    imim = sig_everyn(packed, 2, 1)
    floats_recovered = list(itertools.imap(np.complex, rere, imim))
    return floats_recovered

def raw_to_complex_multi(raw_bytes):
    packed = struct.unpack("%df" % int(len(raw_bytes)/4), raw_bytes)
    rere = sig_everyn(packed, 2, 0)
    imim = sig_everyn(packed, 2, 1)
    floats_recovered = list(itertools.imap(np.complex, rere, imim))
    return floats_recovered

# a pretty-print for hex strings
def get_rose(data):
    try:
        ret = ' '.join("{:02x}".format(ord(c)) for c in data)
    except TypeError, e:
        ret = str(data)
    return ret

def print_rose(data):
    print get_rose(data)

def get_rose_int(data):
    # adding int values
    try:
        ret = ' '.join("{:02}".format(ord(c)) for c in data)
    except TypeError, e:
        ret = str(data)
    return ret

# if you want to go from the pretty print version back to a string (this would not be used in production)
def reverse_rose(input):
    orig2 = ''.join(input.split(' '))
    orig = str(bytearray.fromhex(orig2))
    return orig

# this is meant to replace print comma like functionality for ben being lazy
def s_(*args):
    out = ''
    for arg in args:
        out += str(arg)+' '
    out = out[:-1]
    return out

def print_hex(str, ascii = False):
    print 'hex:'
    tag = ''
    for b in str:
        if ascii:
            if b in string.printable:
                tag = b
            else:
                tag = '?'
        print ' ', format(ord(b), '02x'), tag

def print_dec(str):
    print 'hex:'
    for b in str:
        print ' ', ord(b)

def o_cpm_mod(bits, octave):
    signal = octave.o_cpm_mod(bits, 1/125E1, 1/125E3, 100, 1, sigproto.pattern_vec, 1)
    return signal

def o_cpm_demod(data, octave):
    bits = octave.o_cpm_demod(data, 1/125E3, 100, sigproto.pattern_vec, 1)
    return bits


# http://stackoverflow.com/questions/10237926/convert-string-to-list-of-bits-and-viceversa
def str_to_bits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result

def bits_to_str(bits):
    chars = []
    for b in range(len(bits) / 8):
        byte = bits[b*8:(b+1)*8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)

def all_to_ascii(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(all_to_ascii, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(all_to_ascii, data))
    else:
        return data

def floats_to_bytes(rf):
    return ''.join([complex_to_ishort(x) for x in rf])

def bytes_to_floats(rxbytes):
    packed = struct.unpack("%dh" % int(len(rxbytes)/2), rxbytes)
    rere = sig_everyn(packed, 2, 0)
    imim = sig_everyn(packed, 2, 1)
    assert len(rere) == len(imim)
    rx = list(itertools.imap(np.complex, rere, imim))

# DO NOT USE ROUNDIN GERRORORROROROROOROROORORORO
def drange_DO_NOT_USE(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step



def drange_DO_NOT_USE2(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None:
        inc = 1.0

    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)

    return L


def drange(start, end=None, inc=None):
    """A range function, that does accept float increments..."""
    import math

    if end == None:
        end = start + 0.0
        start = 0.0
    else: start += 0.0 # force it to be a float

    if inc == None:
        inc = 1.0
    count = int(math.ceil((end - start) / inc))

    L = [None,] * count

    L[0] = start
    for i in xrange(1,count):
        L[i] = L[i-1] + inc
    return L




def unroll_angle(input):
    thresh = np.pi

    adjust = 0

    sz = len(input)

    output = [None]*sz

    output[0] = input[0]

    for index in range(1,sz):
        samp = input[index]
        prev = input[index-1]

        if(abs(samp-prev) > thresh):
            direction = 1
            if( samp > prev ):
                direction = -1
            adjust = adjust + 2*np.pi*direction

        output[index] = input[index] + adjust

    return output

def bits_cpm_range(bits):
    bits = [(b*2)-1 for b in bits] # convert to -1,1
    return bits

def bits_binary_range(bits):
    bits = [int((b+1)/2) for b in bits]  # convert to ints with range of 0,1
    return bits

def ip_to_str(address):
    return socket.inet_ntop(socket.AF_INET, address)


# returns None if socket doesn't have any data, otherwise returns a list of bytes
# you need to set os.O_NONBLOCK on the socket at creation in order for this function to work
#   fcntl.fcntl(sock, fcntl.F_SETFL, os.O_NONBLOCK)
def nonblock_socket(sock, size):
    # this try block is the non blocking way to grab UDP bytes
    try:
        buf = sock.recv(size)
    except socket.error, e:
        err = e.args[0]
        if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
            return None  # No data available
        else:
            # a "real" error occurred
            print e
            sys.exit(1)
    else:
        # got data
        return buf

def nonblock_socket_from(sock, size):
    # this try block is the non blocking way to grab UDP bytes
    try:
        buf, addr = sock.recvfrom(size)
    except socket.error, e:
        err = e.args[0]
        if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
            return None, None  # No data available
        else:
            # a "real" error occurred
            print e
            sys.exit(1)
    else:
        # got data
        return buf, addr


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


# write rf to a csv file
# to read file from matlab
# k = csvread('Y:\home\ubuntu\python-osi\qam4.csv');
# kc = k(:,1) + k(:,2)*1j;
def save_rf(filename, data):
    dumpfile = open(filename, 'w')
    for s in data:
        print >> dumpfile, np.real(s), ',', np.imag(s)
    dumpfile.close()

# read rf from a csv file
# if your file is dc in matlab, run this:
#   dcs = [real(dc) imag(dc)];
#   csvwrite('filename.csv',dcs);
#   csvwrite('h3packetshift.csv', [real(dc) imag(dc)]);
def read_rf(filename):
    # read a CSV file
    file = open(filename, 'r')
    reader = csv.reader(file, delimiter=',', quotechar='|')
    data = []
    for row in reader:
        data.append(float(row[0]) + float(row[1])*1j)
    file.close()
    return data


# basic logging setup
def setup_logger(that, name, prefix=None):
    if prefix is None:
        prefix = name

    that.log = logging.getLogger(name)
    that.log.setLevel(logging.INFO)
    # create console handler and set level to debug
    lch = logging.StreamHandler()
    lch.setLevel(logging.INFO)
    lfmt = logging.Formatter(prefix+': %(message)s')
    # add formatter to channel
    lch.setFormatter(lfmt)
    # add ch to logger
    that.log.addHandler(lch)

# convert a list of bits to an unsigned int
# h is the number of bits we are expecting in the list
def bit_list_unsigned_int(lin, h):
    sym = 0
    for j in range(h):
        try:
            sym += lin[j]*2**(h-j-1)
        except IndexError:
            sym += 0
    return sym

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return (idx,array[idx])



octave = None

def get_octave():
    import oct2py
    global octave

    if octave is None:
        print 'starting octave'
        octave = oct2py.Oct2Py()
        # one time add path
        octave.eval("addpath('../prototype/matlab-lib')")
        octave.eval("addpath('../prototype/matlab-ldpc')")
        octave.eval("addpath('../prototype/simulink')")
        octave.eval("pkg load communications")

    return octave

# run this FIRST if you want sigmath to use a client and connect to octave server
def sigmath_octave_use_client(verbose=True):
    import oct2py
    global octave

    if octave is not None:
        print "WARNING OCTAVE ALREADY EXISTED don't call this more than once probably"

    octave = OctaveClient(verbose)  # use zmq client



# ===========================  Test Server idea

_octave_server_url = "ipc://tmp/octave_server0"
_octave_server_url1 = "ipc://tmp/octave_server1"
# _octave_server_url = "inproc://chopperClass"
# _octave_server_url = "inproc://#1"

# never returns
# to run this function just run:
#   python octave_server.py
def run_octave_server():
    self = run_octave_server


    server_print_lim = 25

    # zmq context
    context = zmq.Context.instance()

    zsocketin = context.socket(zmq.SUB)
    zsocketin.setsockopt(zmq.SUBSCRIBE, '') # empty string here subscribes to all channels
    zsocketin.bind(_octave_server_url)

    self.zsocketout = context.socket(zmq.PUB)
    self.zsocketout.bind(_octave_server_url1)

    octave = get_octave()
    octave.eval('who')
    print "Booted, waiting"

    while True:
        try:
            obj = zsocketin.recv_pyobj()

            print "got", str(obj)[0:server_print_lim], "..."

            fn = obj[0]

            arg0 = False
            if len(obj) > 1:
                arg0 = obj[1]

            arg1 = False
            if len(obj) > 2:
                arg1 = obj[2]

            if fn == "boot":
                self.zsocketout.send_pyobj(["bootok"])
            elif fn == "eval":
                octave.eval(arg0)  # no reply
            elif fn == "push":
                octave.push(arg0, arg1)
            elif fn == "pull":
                self.zsocketout.send_pyobj(octave.pull(arg0))
                pass
            else:
                print "Octave Server does not support, '", fn, "'"
        except Exception:
            print Exception
            pass


    # zsocket.connect(worker_url)
    #
    # poller = zmq.Poller()
    # poller.register(worker_routine.zsocket, zmq.POLLIN) # POLLIN for recv, POLLOUT for send


# pretends to be an Octave object, but sends over zmq instead
# things we do with octave
#   octave.eval
#   octave.push
#   octave.pull
#   octave.plot   xxxx
#   octave.real   xxxx
#   octave.imag   xxxx

class OctaveClient(object):
    def __init__(self, verbose=True):
        self.context = zmq.Context.instance()
        self.zsocketout = self.context.socket(zmq.PUB)
        self.zsocketout.connect(_octave_server_url)

        self.zsocketin = self.context.socket(zmq.SUB)
        self.zsocketin.setsockopt(zmq.SUBSCRIBE, '')
        self.zsocketin.connect(_octave_server_url1)

        time.sleep(10/1000.0)  # mandatory sleep after connect

        self.send_boot()

        self.print_lim = 50
        self.verbose = verbose

    def send_boot(self):
        self.zsocketout.send_pyobj(['boot'])
        print "TRYING TO CONNECT TO OCTAVE...."
        boot_back = self.zsocketin.recv_pyobj()
        print "CONNECTED!!!"
        assert(boot_back == ["bootok"])

    def eval(self, text):
        if self.verbose:
            print "Client running:", text[0:self.print_lim]
        self.zsocketout.send_pyobj(['eval', text])
        # boot_back = self.zsocketin.recv_pyobj()
        # assert(boot_back == ["bootok"])

    def push(self, name, obj):
        if self.verbose:
            print "Client pushing:", name
        self.zsocketout.send_pyobj(['push', name, obj])

    def pull(self, name):
        if self.verbose:
            print "Client pulling:", name
        self.zsocketout.send_pyobj(['pull', name])
        pulled = self.zsocketin.recv_pyobj()
        return pulled



def get_octave_via_server():
    c = OctaveClient()
    return c



def oplot(mod):
    octave = get_octave()

    octave.eval("figure")
    octave.plot(octave.imag(mod))

def oplotxy(mod):
    octave = get_octave()

    octave.eval("figure")
    octave.plot(mod)

def oplotr(mod):
    octave = get_octave()

    octave.eval("figure")
    octave.plot(octave.real(mod))

def osplot(mod):
    octave = get_octave()
    octave.eval("figure")
    octave.push('osplotd1', list(mod))
    octave.eval("splot(osplotd1.')")
    octave.eval('clear osplotd1')

def oplotqam(mod):
    octave = get_octave()
    octave.eval("figure")
    octave.push('oplotqamd1', list(mod))
    octave.eval("plot(oplotqamd1,'.b');")
    octave.eval('clear oplotqamd1')


def oplotf(mod, fs):
    octave = get_octave()

    octave.eval("figure")
    # octave.plot(np.imag(mod))
    octave.push('oplotf1', list(mod))
    octave.push('oplotf2', fs)
    octave.eval("fplot(oplotf1, oplotf2)")
    octave.eval('clear oplotf1 oplotf2')

def interpxx(data):
    octave = get_octave()

    octave.push('interp5data', data)
    # octave.eval('interp5output = interp5data;')
    octave.eval('interp5output = interpmatlab(interp5data, 5, 4, 0.01 );')
    dout = octave.pull('interp5output')
    octave.eval('clear interp5data interp5output')

    return dout[0]


def interp6(data):
    octave = get_octave()

    octave.push('interp5data', data)
    # octave.eval('interp5output = interp5data;')
    octave.eval('interp5output = interpmatlab(interp5data, 6, 1, 0.1 );')
    dout = octave.pull('interp5output')
    octave.eval('clear interp5data interp5output')

    return dout[0]

def interpn(data, R, N, alpha):
    octave = get_octave()

    octave.push('interp5data', data)
    octave.push('interp5dataR', R)
    octave.push('interp5dataN', N)
    octave.push('interp5dataalpha', alpha)
    # octave.eval('interp5output = interp5data;')
    octave.eval('interp5output = interpmatlab(interp5data, interp5dataR, interp5dataN, interp5dataalpha );')
    dout = octave.pull('interp5output')
    octave.eval('clear interp5data interp5dataR interp5dataN interp5dataalpha interp5output')

    return dout[0]


def freq_shift(data, fs, shift):
    octave = get_octave()

    octave.push('octinput1', data)
    octave.push('octinput2', fs)
    octave.push('octinput3', shift)
    # octave.eval('interp5output = interp5data;')
    octave.eval("octoutput = freq_shift(octinput1, octinput2, octinput3);")
    # octave.eval("disp(octoutput)")
    dout = octave.pull('octoutput')
    octave.eval('clear octinput1 octinput2 octinput3 octoutput')

    return np.array(list(sigflatten(dout)))

def tone_gen(samples, fs, hz):
    inc = 1.0/fs * 2 * np.pi * hz

    if hz == 0:
        args = np.array([0] * samples)
    else:
        args = np.arange(0,samples*inc,inc)*np.array([1j]*samples)
    return np.exp(args)

# This doesn't match the accuracy of the octave version
# def freq_shift(din, fs, shift):
#     sz = len(din)
#     sampleInc = 1.0/fs * 2.0 * np.pi * shift
#     endSample = (sz/fs) * 2.0 * np.pi * shift
#     # tvec = drange(0, sampleInc, endSample)
#     tvec = []
#     # print 0, endSample, sampleInc
#     for x in drange(0, endSample, sampleInc):
#         tvec.append(x)
#
#     tvec = np.array(tvec)
#
#     # print tvec
#     # tvec = 0:sampleInc:endSample;
#
#     shiftTone = np.sin(tvec) + 1j*np.cos(tvec)
#
#     dout = shiftTone * din
#     oplotr(shiftTone)
#     oplotr(din)
#     oplotr(dout)
#     return dout


# sinc interpolate data at n evently spaced pointes
# for upsample by 3 use len(data)*3
def interpft(data, N):
    octave = get_octave()

    octave.push('interpftdata', data)
    octave.push('interpftN', N)
    # octave.eval('interp5output = interp5data;')
    octave.eval('interpftoutput = interpft(interpftdata, interpftN);')
    dout = octave.pull('interpftoutput')
    octave.eval('clear interpftdata interpftN interpftoutput')

    return dout[0]







# ---------------- Still Octave, but LDPC ----------------
# these scripts require that python-osi and prototype repos are checked out at the same time
# for instance:
#   /home/ubuntu/python-osi
#   /home/ubuntu/prototype

def ldpcgen(n, k, onesmin=1, onesmax=2):
    octave = get_octave()

    octave.push('octinput1', n)
    octave.push('octinput2', k)
    octave.push('octinput3', onesmin)
    octave.push('octinput4', onesmax)
    octave.eval('octoutput1 = ldpcgen(octinput1, octinput2, octinput3, octinput4);')
    # octave.eval('spy(octoutput1)')
    dout = octave.pull('octoutput1')
    octave.eval('clear octinput1 octinput2 octinput3 octinput4 octoutput1')

    return dout

def o_ldpcencode(G, u):
    octave = get_octave()

    octave.push('octinput1', G)
    octave.push('octinput2', u)
    octave.eval("octoutput1 = ldpcencode(octinput1, octinput2);")
    dout = octave.pull('octoutput1')
    octave.eval('clear octinput1 octinput2 octoutput1')

    return dout[0]

def ldpcencode(G, u):
    return np.dot(u, G) % 2

def ldpcencodesparse(G, u):
    sparse = u.dot(G)
    gf2 = sparse.todense() % 2
    ndarray = np.array(gf2)[0]
    return ndarray

def ldpcpar2gen(H):
    octave = get_octave()

    octave.push('octinput1', H)
    octave.eval('octoutput1 = ldpcpar2gen(octinput1);')
    dout = octave.pull('octoutput1')
    octave.eval('clear octinput1 octinput2 octoutput1')

    return dout

def gen2par(G):
    octave = get_octave()

    octave.push('octinput1', G)
    octave.eval('octoutput1 = gen2par(octinput1);')
    dout = octave.pull('octoutput1')
    octave.eval('clear octinput1 octinput2 octoutput1')

    return dout

def g2rref(G):
    octave = get_octave()

    octave.push('octinput1', G)
    octave.eval('octoutput1 = g2rref(octinput1);')
    dout = octave.pull('octoutput1')
    octave.eval('clear octinput1 octinput2 octoutput1')

    return dout

def mat_nicename(H):
    return "Matrix (h,w) %d %d"%(H.shape[0],H.shape[1])

# returns
def ldpccheckHGOrthogonal(H, G):
    res = (np.dot(G,np.transpose(H)))

    res = res % 2  # always required after math in GF(2)

    print "res", mat_nicename(res)

    if np.sum(res) == 0:
        return True
    else:
        return False

# returns the syndrome
def ldpccheck(H, cw):
    octave = get_octave()

    octave.eval("addpath('../prototype/matlab-lib')")
    octave.eval("addpath('../prototype/matlab-ldpc')")

    octave.push('octinput1', H)
    octave.push('octinput2', cw)
    octave.eval("octoutput1 = ldpccheck(octinput1, octinput2).';")
    dout = octave.pull('octoutput1')
    octave.eval('clear octinput1 octinput2 octoutput1')

    return dout[0]


def ldpcHGInfo(H, G, verify = True):
    n = H.shape[1]
    k = n - H.shape[0]

    problems = False

    assert G.shape[0] == k
    assert G.shape[1] == n

    if not verify:
        return (n, k, problems)

    # k codeword length
    # n original message length

    ortho = ldpccheckHGOrthogonal(H, G)

    if not ortho:
        print "WARNING NOT ORTHO"
        problems = True

    if not np.array_equal(np.eye(k,k), G[:,0:k]):
        problems = True
        print "WARNING G is NOT a systematic"  # code retreival will not work

    # assert ortho, "Not ortho"

    return (n, k, problems)

def ldpcPrintHGInfo(H, G):
    (n,k,problems) = ldpcHGInfo(H,G)

    print "H", mat_nicename(H)
    print "G", mat_nicename(G)
    print "together these guys have"
    print "n", n
    print "k", k
    print "this makes a rate k/n", float(k)/n

# inplace modifies H
def mat_switch_rows(H, ida, idb):
    t = np.copy(H[ida,:])
    H[ida,:] = H[idb,:]
    H[idb,:] = t
    return None

# inplace
def mat_switch_cols(H, ida, idb):
    t = np.copy(H[:,ida])
    H[:,ida] = H[:,idb]
    H[:,idb] = t
    return None


def o_ldpc_load_h(H):
    octave = get_octave()

    octave.push('o_ldpc_H', H)


def o_ldpc_par2gen(H):
    octave = get_octave()

    octave.push('octinput1', H)
    code = [
        "t1 = gf(octinput1,1)",
        "[par_bits,tot_bits] = size(t1)",
        "inf_bits = tot_bits - par_bits",
        "tH1 = t1(:,1:inf_bits)",
        "tH2 = t1(:,inf_bits+1:end)",
        "t3p = (inv(tH2)*(-tH1))'",
        "tG = [ gf(eye(inf_bits),1) t3p ]",
        "tGd = double(tG.x)"
    ]
    for x in code:
        octave.eval(x+";")
    dout = octave.pull("tGd")
    return dout

def o_ldpc_load_matlab_edu(path):
    octave = get_octave()

    octave.eval("load " + path)

    # octave.eval('who')
    octave.eval('size(LDPC.H.x)')
    octave.eval('hh = gf(LDPC.H.x,1);')
    octave.eval('hh2 = int32(hh.x);')

    octave.eval('gg = gf(LDPC.G.x,1);')
    octave.eval('gg2 = int32(gg.x);')

    H = octave.pull('hh2')
    G = octave.pull('gg2')

    octave.eval('clear LDPC hh hh2 gg gg2')
    return H, G


def o_xcr_downsample_preup(data, combup, upfactor, downfactor, testing=0):
    octave = get_octave()
    octave.push('xdp_data', data)
    if combup is not False:
        octave.push('xdp_combup', combup)
    octave.push('xdp_i1', upfactor)
    octave.push('xdp_i2', downfactor)
    octave.push('xdp_i3', testing)
    octave.eval("xdp_o1 = xcr_downsample_preup(xdp_data.', xdp_combup.', xdp_i1, xdp_i2, xdp_i3).';")
    perfectdata = octave.pull("xdp_o1")
    octave.eval('clear xdp_data xdp_i1 xdp_i2 xdp_o1')  # don't clear the combup

    return perfectdata[0]

def o_xcr_downsample_freq_preup(data, upfactor, downfactor):
    octave = get_octave()
    octave.push('xdp_data', data)
    octave.push('xdp_i1', upfactor)
    octave.push('xdp_i2', downfactor)
    octave.eval("xdp_o1 = xcr_downsample_freq_preup(xdp_data.', xdp_combup.', xdp_combup_end.', xdp_i1, xdp_i2).';")
    perfectdata = octave.pull("xdp_o1")
    octave.eval('clear xdp_data xdp_i1 xdp_i2 xdp_o1')  # don't clear the combup

    return perfectdata[0]



def o_xcr_freq_detect_preload(comb, fs):
    octave = get_octave()


#     the pushed names here CANNOT be changed without editing the .m file
    octave.push('xcr_freq_detect_input_comb', comb)
    octave.push('xcr_freq_detect_fs', fs)
    octave.eval("xcr_freq_detect_input_comb = xcr_freq_detect_input_comb.';")
    octave.eval('xcr_freq_detect_preload')

# returns how far the wave is freq shifted
# take the negative of this value and apply to freq_shift
def o_xcr_freq_detect(data):
    octave = get_octave()

    # This name can be whatever, we hold this on purpose
    octave.push('xcr_data_hold', data)
    octave.eval("xcr_data_hold = xcr_data_hold.';")

    octave.eval('xcr_detected_offset = xcr_freq_detect(xcr_data_hold, xcr_freq_detect_range, xcr_freq_combs, xcr_freq_detect_fs);')
    offset = octave.pull('xcr_detected_offset')

    # octave.eval('clear xcr_i1')

    return offset


def o_phy_rx_magic_preload(combup, combupend):
    octave.push('xdp_combup', combup)
    octave.push('xdp_combup_end', combupend)

# this function is really dangerous and not DRY but is faster by avoiding pushing and pulling data extra times
# must call o_xcr_downsample_preup with valid comb first
def o_phy_rx_magic(data, upfactor, downfactor, testing=0):
    octave = get_octave()

    octave.push('xcr_data_hold', data)
    octave.push('xdp_i1', upfactor)
    octave.push('xdp_i2', downfactor)
    octave.push('xdp_i3', testing)

    commands = [
        "xcr_data_hold = xcr_data_hold.'",
        "xcr_detected_offset = xcr_freq_detect(xcr_data_hold, xcr_freq_detect_range, xcr_freq_combs, xcr_freq_detect_fs)",
        "phy_magic_shifted = freq_shift(xcr_data_hold, xcr_freq_detect_fs, -1*xcr_detected_offset)",
        "xdp_o1 = xcr_downsample_preup(phy_magic_shifted, xdp_combup.', xdp_i1, xdp_i2, xdp_i3).'",
        ''] # final empty string provides the final ;

    lump_string = ';'.join(commands)
    octave.eval(lump_string)
    perfectdata = octave.pull("xdp_o1")
    octave.eval('clear xdp_data xdp_i1 xdp_i2 xdp_o1')

    return perfectdata[0]

def o_phy_not_rx_magic(rx, upfactor, downfactor):
    fs = (1E8/512/2) * 3
    offset = o_xcr_freq_detect(rx)
    # print "found offset of", offset

    # shift by the negative
    qamshifted = freq_shift(rx, fs, -1*offset)

    perfectrf = o_xcr_downsample_freq_preup(qamshifted, 11, 3)
    return perfectrf



# ---------------- End of Octave ----------------

# pass in any sparse, this will convert to bsr sparse and save
def fancy_save_sparse(filename, H):
    Hsparse = scipy.sparse.bsr_matrix(H, dtype=np.int16)
    save_sparse_csr(filename, Hsparse)

# def fancy_load_sparse(filename):
#     loader = np.load(filename)
#     bsr = scipy.sparse.bsr_matrix((  loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])


# only pass type of bsr
def save_sparse_csr(filename, array):
    assert type(array) == scipy.sparse.bsr.bsr_matrix
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.bsr_matrix((  loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])


# import scipy.sparse as sps
def nplotspy(A, title=""):
    global _nplot_figure
    plt.figure(_nplot_figure)

    plt.title(title)
    plt.spy(A, precision=0.01, markersize=3)

    _nplot_figure += 1














def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N



# ---------------- Beginning of matplotlib

_nplot_figure = 0
def nplot(rf, title="", hold=False):
    global _nplot_figure

    if hold:
        plt.hold(True)
    else:
        plt.figure(_nplot_figure)
        plt.title(title)
    plt.plot(rf)
    # plt.plot(range(len(rf)), np.real(rf), '-ko')
    _nplot_figure += 1

def nplotdots(rf, title="", hold=False):
    global _nplot_figure

    if hold:
        plt.hold(True)
    else:
        plt.figure(_nplot_figure)
        plt.title(title)
    # plt.plot(rf)
    plt.plot(range(len(rf)), np.real(rf), '-ko')
    _nplot_figure += 1



def nplotqam(rf, title="", hold=False):
    global _nplot_figure
    plt.figure(_nplot_figure)
    plt.title(title)
    plt.plot(np.real(rf), np.imag(rf), '.b', alpha=0.6)
    # plt.plot(rf)
    _nplot_figure += 1


def nplotfft(rf, title=""):
    global _nplot_figure
    plt.figure(_nplot_figure)
    plt.title(title)
    # plt.figure(3)
    plt.plot(abs(fftshift(fft(rf))))
    _nplot_figure += 1

def nplothist(x, title = "", bins = False):
    global _nplot_figure
    plt.figure(_nplot_figure)
    _nplot_figure += 1

    plt.title(title)

    # more options
    # n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

    if bins is not False:
        plt.hist(x, bins)
    else:
        plt.hist(x)

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)


def nplotber(bers, ebn0, titles, title = ""):
    global _nplot_figure

    assert(len(bers) == len(ebn0) == len(titles))

    plt.figure(_nplot_figure)
    _nplot_figure += 1

    maintitle = "BER of ("

    for i in range(len(bers)):
        plt.semilogy(ebn0[i], bers[i], '-s', linewidth=2)
        maintitle += titles[i] + ', '

    maintitle += ")"



    gridcolor = '#B0B0B0'
    plt.grid(b=True, which='major', color=gridcolor, linestyle='-')
    plt.grid(b=True, which='minor', color=gridcolor, linestyle='dotted')

    plt.legend(titles)
    plt.xlabel('Eb/No (dB)')
    plt.ylabel('BER')

    if title == "":
        plt.title(maintitle)
    else:
        plt.title(title)


def nplotshow():
    plt.show()







def rand_string_ascii(len):
    return ''.join(random.choice(string.ascii_letters) for _ in range(len))



def sigflatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def sigdup(listt, count=2):
    # return list(flatten([y for y in repeat([y for y in listt], count)]))
    return list(sigflatten([repeat(x, count) for x in listt]))




def sig_fft(data, Fs):

    dt = 1/Fs;     #                % seconds per sample
    N = len(data);
     # Fourier Transform:
    X = fftshift(fft(data)/N);
    # %% Frequency specifications:
    dF = float(Fs)/N;      #                % hertz
    # f = -Fs/2:dF:Fs/2-dF;#           % hertz
    f = drange(-Fs/2, Fs/2, dF)

    return X, f
    # %% Plot the spectrum:
    # figure;
    # plot(f,2*abs(X));
    # xlabel('Frequency (in hertz)');
    # title('Magnitude Response');
    # disp('Each bin is ');
    # disp(dF);


def sig_rms(x):
    return np.sqrt(np.mean(x * np.conj(x)))



# returns the FIRST occurrence of the maximum
# this is the fastest by FAR, but it requires that data be an ndarray
# behaves like matlab max() do not pass in complex values and assume that this works
def sig_max(data):
    idx = np.argmax(data)
    m = data[idx]
    return (m, idx)

def sig_max2(data):
    m = max(data)
    idx = [i for i, j in enumerate(data) if j == m]
    idx = idx[0]
    return (m, idx)

def sig_max3(data):
    m = max(data)
    idx = data.index(m)
    return (m, idx)

def sig_max4(data):
    m = max(data)
    idx = data.index(m)
    return (m, idx)


def sig_everyn(data, n, phase=0):
    l = len(data)
    assert phase < n, "Phase must be less than n"
    assert n <= l, "n must be less than or equal length of data"

    # subtract the phase from length, and then grab that many elements from the end of data
    # this is the same as removing 'phase' elements from the beginning
    if phase != 0:
        phaseapplied = data[-1*(l-phase):]

        # after applying phase, return every nth element from that
        return phaseapplied[::n]
    else:
        return data[::n]

# http://stackoverflow.com/questions/17904097/python-difference-between-two-strings/17904977#17904977

def sig_diff(a,b,max=False):
    at = time.time()
    count = 0
    # print('{} => {}'.format(a,b))
    for i,s in enumerate(difflib.ndiff(a, b)):
        if s[0]==' ': continue
        elif s[0]=='-':
            try:
                c = u'{}'.format(s[-1])
            except:
                c = ' '
            print(u'Delete "{}" from position {}'.format(c,i))
        elif s[0]=='+':
            print(u'Add "{}" to position {}'.format(s[-1],i))
        count += 1
        if max and count >= max:
            print "stopping after", count, "differences"
            break
    print()
    bt = time.time()
    print "sig_diff() ran in ", bt-at

def sig_diff2(a, b, unused=False):
    lmin = min(len(a),len(b))
    lmax = max(len(a),len(b))

    errors = 0
    for i in range(lmin):
        if a[i] != b[i]:
            errors += 1

    errors += lmax-lmin
    if lmax != lmin:
        print "Strings were", errors, "different with additional", lmax-lmin, "due to length differences"
    else:
        print "Strings were", errors, "different"

# uses ndimage.label() with a threshold to detect features, count the features, and then
# count the average number of pixels per feature
def sig_thresh_label(data, thresh, show=False):
        labeled, nr_objects = ndimage.label(data > thresh)
        # print "found", nr_objects, "objects with thresh", thresh
        label_indices = [(labeled == i).nonzero() for i in xrange(1, nr_objects+1)]

        assert nr_objects == len(label_indices)

        pixel_total = 0

        for j in range(nr_objects):
            label = label_indices[j]
            c = len(label[0])
            pixel_total += c
            # print "lable", j, "has", c, "pixels"
        pixel_count_average = pixel_total / float(nr_objects)
        # print "avrage", pixel_count_average

        # after all math done, optionally show it
        if show:
            plt.imshow(labeled)
            io.show()

        return nr_objects, pixel_count_average

# most inefficient thing ever, actually writes to a file in order to group points on a qam plot
def sig_plot_group(points, thresh):

    # temp file used for this process (can we use ram somehow (worsecase, a ramdisk?))
    filename = "tmp/sig_plot_group.png"

    plt.figure(figsize=(1,1), dpi=800)
    plt.axis('off')
    plt.plot(np.real(points), np.imag(points), '.b', alpha=0.4)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=400)
    plt.close()

    image = Image.open(filename)
    gray = image.convert('L')
    grayar = np.asarray(gray)
    grayar = np.invert(grayar)
    return sig_thresh_label(grayar, thresh)

# builds a ring of evenly spaced points, used for circular QAM
def buildQAMRing(r, count, rot=0):
        ainc = 1.0/count * 2 * np.pi

        points = [0] * count

        for i in range(count):
            p = np.e ** (1j * (rot + ainc*i))
            p *= r  # scale to amplitude for the ring
            points[i] = p
        return points

# returns the closest distance between a list of complex points
# used for checking the quality of QAM constellations
def listClosestDistance(points, verbose=True):

        best = 999999
        closestpair = (-1,-1)

        for i in range(len(points)):
            pouter = points[i]
            for k in range(len(points)):
                if i == k:
                    continue
                pinner = points[k]
                diff = min(best, abs(pouter-pinner))
                if diff < best:
                    best = diff
                    closestpair = (pouter,pinner)
        if verbose:
            print "closest", best, closestpair

        return (best,closestpair)

def sig_awgn(data, snrdb):
    ll = len(data)

    # only way I know to make complex noise
    noise = np.random.normal(0, 1, ll) + 0j
    rot = np.random.random_sample(ll)
    for i in range(len(noise)):
        noise[i] = noise[i] * np.exp(1j*rot[i]*2.0*np.pi) # random rotation with same amplitude

    sigma2 = 10.0**(-snrdb/10.0)
    out = data + noise*math.sqrt(sigma2)

    return out

# accepts numpy matrices, always uses int16 math to compute sha
def sig_sha256_matrix(H):
    marshal = np.int16(H).tolist()
    return sig_sha256(marshal)

# converts the sparse matrix into dense, then returns the sha256
def sig_sha256_sparse_matrix(H):
    return sig_sha256_matrix(H.todense())

def sig_sha256(as_str):
    worker = hashlib.sha256()
    worker.update(str(as_str))
    return worker.digest()


def get_chopper_work_url():
    return "ipc://tmp/chopper_work"
def get_chopper_control_url():
    return "ipc://tmp/chopper_control"
def get_chopper_feedback_url():
    return "ipc://tmp/chopper_feedback"

# def sinc_interp(x, s, u):
#     """
#     Interpolates x, sampled at "s" instants
#     Output y is sampled at "u" instants ("u" for "upsampled")
#
#     from Matlab:
#     http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
#     """
#
#     if len(x) != len(s):
#         raise Exception, 'x and s must be the same length'
#
#     # Find the period
#     T = s[1] - s[0]
#
#
#     A = np.tile(u, (len(s), 1))
#     B = s[:, np.newaxis]
#
#
#     sincM = A - np.tile(B, (1, len(u)))
#     y = np.dot(x, np.sinc(sincM/T))
#     return y
