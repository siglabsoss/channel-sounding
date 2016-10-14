
from subprocess import call





def make_hz_patch(name, hz):
    #base = '33c33\n<         self.samp_freq = samp_freq = 915E6\n---\n>         self.samp_freq = samp_freq = 916E6\n'
    base = '33c33\n<         self.samp_freq = samp_freq = 915E6\n---\n>         self.samp_freq = samp_freq = '
    final = base + str(hz) + '\n'
    fo = open(name, 'w')
    fo.write(final)
    fo.close()

def add_target(makebase, filename, basefile, patches):

    makebase = makebase + filename + ': ' + basefile + '\n'
    makebase = makebase + '\tcp ' + basefile + ' ' + filename + '\n'
    for p in patches:
        makebase = makebase + '\tpatch ' + filename + ' < ' + p + '\n'

    return makebase + '\n\n'

def mkfname(hzn):
    return str(hzn) + '_patch.patch'

if __name__ == '__main__':
    
    # First step, copy the makefile over
    fo = open('Makefile.base', 'r')
    makebase = fo.read(999999)
    fo.close()



    hzname = ['905E6', '910E6', '915E6', '920E6', '923E6']

    # types = ['r', 't']
    types = ['r']
    base_grc = ['drive_rx.py']


    make_hz_patch(mkfname('905E6'), 905E6)  # write a patch to the filesystem
    makebase = add_target(makebase, '_rx_905E6.py', base_grc[0], [mkfname('905E6')]) # include it in make
    call(['rm', 'Makefile'])
    makefo = open('Makefile', 'w')
    makefo.write(makebase)
    makefo.close()

    print makebase

    # prefix 
    
    # f = open("change_freq.patch", "r")
    # s = f.read(999)
    # print repr(s)

    for i in range(len(types)):
        type = types[i]
        infile = base_grc[i]
        outbase = '_' + type + 'x' + '_'
        # call['tp']
        for name in hzname:
            hz = eval(name)
            outfile = outbase + name + '.py'
            # call(['cp', infile, outfile])
            # print "final file: " + outfile
        #     print hz
        #     print name
            # make_hz_patch(mkfname(name), hz)

        # call(['ls'])
    # jak = initSerial('/dev/ttyUSB0', 115200)
    # print jak