from subprocess import call


def make_hz_patch(name, hz):
    #base = '33c33\n<         self.samp_freq = samp_freq = 915E6\n---\n>         self.samp_freq = samp_freq = 916E6\n'
    base = '33c33\n<         self.samp_freq = samp_freq = 915E6\n---\n>         self.samp_freq = samp_freq = '
    final = base + str(hz) + '\n'
    fo = open(name, 'w')
    fo.write(final)
    fo.close()

def mkfname(hzn):
	return str(hzn) + '_patch.patch'

if __name__ == '__main__':
    hzname = ['915E6', '917E6']
    
    # f = open("change_freq.patch", "r")
    # s = f.read(999)
    # print repr(s)


    for hzn in hzname:
    	hz = eval(hzn)
        print hz
        print hzn
        # make_hz_patch(mkfname(hzn), hz)

    call(['ls'])
    # jak = initSerial('/dev/ttyUSB0', 115200)
    # print jak