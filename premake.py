
from subprocess import call
import argparse





class MakeWrap:
    makebasefn = 'Makefile.base'
    outputs = []
    # self.makebase

    def init(self, o):
        fo = open(self.makebasefn, 'r')
        self.makebase = fo.read(999999)
        fo.close()
        self.output_folder = o

    def add_line(self, ln):
        self.makebase = self.makebase + ln

    def add_target(self, filename, basefile, patches):
        self.add_line(filename + ': ' + basefile + '\n')
        self.add_line('\tcp ' + basefile + ' ' + filename + '\n')
        for p in patches:
            self.add_line('\tpatch ' + filename + ' < ' + p + '\n')

        self.add_line('\n\n')

        self.outputs.append(filename)


    def delme(self):
        make_hz_patch(mkfname('905E6'), 905E6)  # write a patch to the filesystem
        self.add_target('_rx_905E6.py', base_grc[0], [mkfname('905E6')])  # include it in make

    def finalize(self):

        targets = ""
        delim = ""
        for t in self.outputs:
            targets = targets + delim + t
            delim = " "

        self.add_line('runrx: ' + targets + '\n')
        self.add_line('\tsudo ls > /dev/null\n')
        for t in self.outputs:
            self.add_line('\tsudo ./' + t + '\n')

        self.add_line("\t@sudo touch " + self.output_folder + '/"`date`"\n')
        self.add_line("\t@echo made these files\n")
        self.add_line("\t@echo \n")
        self.add_line("\tls -lsh " + self.output_folder + '\n')

        self.add_line("\n.PHONY: runrx\n")

        self.add_line('\n\n')



        call(['rm', '-f', 'Makefile'])
        makefo = open('Makefile', 'w')
        makefo.write("all : " + targets + '\n\n' + self.makebase)
        makefo.close()


def make_hz_patch(name, hz):
    #base = '33c33\n<         self.samp_freq = samp_freq = 915E6\n---\n>         self.samp_freq = samp_freq = 916E6\n'
    base = '33c33\n<         self.samp_freq = samp_freq = 0\n---\n>         self.samp_freq = samp_freq = '
    final = base + str(hz) + '\n'
    fo = open(name, 'w')
    fo.write(final)
    fo.close()


def mkfname(hzn):
    return '_patch_' + str(hzn) + '.patch'

def mkrawname(hzn, test_name):
    return test_name + '_' + hzn + '.raw'

def mkpathpatchname(hzn, test_name):
    return '_patch_output_' + test_name + '_' + hzn + '.path'

def make_filename_patch(name, output):
    # base = '31c31\n<         self.output_file = output_file = ""\n---\n>         self.output_file = output_file = "/mnt/usb1/foldername/filename.raw"\n'
    base = '31c31\n<         self.output_file = output_file = ""\n---\n>         self.output_file = output_file = '
    final = base + '"' + output + '"\n'

    fo = open(name, 'w')
    fo.write(final)
    fo.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Prepare Siglabs Suitcase tx/rx Makefile generator")
    parser.add_argument('name', nargs='+', help='Pass in the name of the test to be run.  Output will be in a folder named this')
    # parser.add_argument('positional', nargs="+", help="A positional argument")
    # parser.add_argument('--optional', help="An optional argument")

    args = parser.parse_args()

    output_base = '/mnt/usb1'
    test_name = vars(args)['name'][1]

    print repr(test_name)
    output_folder = output_base + '/' + 'test_' + test_name

    call(['sudo', 'mkdir', '-p', output_folder])

    
    # First step, copy the makefile over


    hzname = ['905E6', '910E6', '915E6', '920E6', '923E6']
    hzname = ['905E6', '910E6']

    # types = ['r', 't']
    types = ['r']
    base_grc = ['drive_rx.py']

    wrap = MakeWrap()

    wrap.init(output_folder)



    for i in range(len(types)):
        type = types[i]
        infile = base_grc[i]
        outbase = '_' + type + 'x' + '_'
        # call['tp']
        for name in hzname:
            print "processing " + type + "x for hz " + name
            hz = eval(name)
            outfile = outbase + name + '.py'

            # patch the hz
            hzpatchname = mkfname(name)
            make_hz_patch(hzpatchname, hz)

            # patch the path
            rawpath = mkrawname(name, test_name)
            pathpatchname = mkpathpatchname(name, test_name)
            full_output = output_folder + '/' + rawpath
            make_filename_patch(pathpatchname, full_output)

            # setup makefile
            wrap.add_target(outfile, infile, [hzpatchname, pathpatchname, 'sleep4.patch'])  # let make know about it

            print pathpatchname


    wrap.finalize()
