
from subprocess import call
import argparse





class MakeWrap:
    makebasefn = 'Makefile.base'
    outputs = {'t':[],'r':[]}

    def init(self, o):
        fo = open(self.makebasefn, 'r')
        self.makebase = fo.read(999999)
        fo.close()
        self.output_folder = o

    def add_line(self, ln):
        self.makebase = self.makebase + ln

    def add_target(self, txrx, filename, basefile, patches):
        self.add_line(filename + ': ' + basefile + ' _premake_run\n')
        self.add_line('\tcp ' + basefile + ' ' + filename + '\n')
        for p in patches:
            self.add_line('\tpatch ' + filename + ' < ' + p + '\n')

        self.add_line('\n\n')

        self.outputs[txrx].append(filename)

    def finalize(self):

        types = ['r', 't']


        for type in types:
            if type == 't':
                runnerfo = open('_tx_targets', 'w')
            if type == 'r':
                runnerfo = open('_rx_targets', 'w')

            runnerfo.write(" ".join(self.outputs[type]))
            runnerfo.close()


        alltargets = ""

        for type in types:
            targets = ""
            for targ in self.outputs[type]:
                targets = targets + " " + targ + " "
                alltargets = alltargets + " " + targ + " "

            self.add_line('run' + type + 'x: ' + targets + '\n')
            self.add_line('\tsudo ls > /dev/null\n')
            self.add_line('\tsudo python ./runner.py type ' + type + 'x\n')

            if type == 'r':
                self.add_line("\t@sudo touch " + self.output_folder + '/"`date`"\n')
                self.add_line("\t@echo made these files\n")
                self.add_line("\t@echo \n")
                self.add_line("\tls -lsh " + self.output_folder + '\n')
            self.add_line('\n\n')

        self.add_line("\n.PHONY: runrx runtx\n")
        self.add_line('\n\n')



        call(['rm', '-f', 'Makefile'])
        makefo = open('Makefile', 'w')
        makefo.write("all : drive_rx.py drive_tx.py " + alltargets + '\n\n' + self.makebase)
        makefo.close()


def mkgainpatchname(gain):
    return '_patch_gain_' + str(gain) + '.patch'

def make_gain_patch(name, gain):
    base = '27c27\n<         self.txrx_gain = txrx_gain = -1\n---\n>         self.txrx_gain = txrx_gain = '
    final = base + str(gain) + '\n'
    fo = open(name, 'w')
    fo.write(final)
    fo.close()

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
    return '_patch_output_' + test_name + '_' + hzn + '.patch'

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

    args = parser.parse_args()

    output_base = '/mnt/usb1'
    test_name = vars(args)['name'][1]

    # print repr(test_name)
    output_folder = output_base + '/' + 'test_' + test_name

    call(['sudo', 'mkdir', '-p', output_folder])

    # this file lets make know if premake was run since targets were last made
    call(['touch', '_premake_run'])

    
    # First step, copy the makefile over


    # hzname = ['905E6', '910E6', '915E6', '2410E6', '2420E6', '2430E6']
    hzname = {'9':['905E6', '910E6', '915E6'], '2.4':['2410E6', '2420E6', '2430E6'] }

    types = ['r', 't']
    bands = ['9', '2.4']

    gains = {'9': {'t': 5, 'r': 3}, '2.4': {'t': 24, 'r': 10}}



    base_grc = ['drive_rx.py', 'drive_tx.py']

    wrap = MakeWrap()

    wrap.init(output_folder)



    for i in range(len(types)):
        type = types[i]
        infile = base_grc[i]
        outbase = '_' + type + 'x' + '_'
        for band in bands:
            for name in hzname[band]:
                print "processing " + type + "x for hz " + name
                hz = eval(name)
                outfile = outbase + name + '.py'

                # patch the hz
                hzpatchname = mkfname(name)
                make_hz_patch(hzpatchname, hz)

                if type == 'r':
                    patch_list = [hzpatchname, 'sleep62.patch']

                    # patch the path
                    rawpath = mkrawname(name, test_name)
                    pathpatchname = mkpathpatchname(name, test_name)
                    full_output = output_folder + '/' + rawpath
                    make_filename_patch(pathpatchname, full_output)
                    patch_list.append(pathpatchname)

                    # patch the gain
                    rxgain = gains[band][type]
                    gainpatchname = mkgainpatchname(rxgain)
                    make_gain_patch(gainpatchname, rxgain)
                    patch_list.append(gainpatchname)

                    if band == '2.4':
                        patch_list.append('change_antenna.patch') # patch antenna only for rx in 2.4G


                    # setup makefile
                    wrap.add_target(type, outfile, infile, patch_list)  # let make know about it
                if type == 't':

                    # patch the gain
                    txgain = gains[band][type]
                    gainpatchname = mkgainpatchname(txgain)
                    make_gain_patch(gainpatchname, txgain)

                    wrap.add_target(type, outfile, infile, [hzpatchname, gainpatchname, 'sleep60.patch'])  # let make know about it



    wrap.finalize()