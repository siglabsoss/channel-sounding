#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Cal Rx
# Generated: Thu Oct 13 19:07:26 2016
##################################################

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import time


class cal_rx(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Cal Rx")

        ##################################################
        # Variables
        ##################################################
        self.txrx_gain = txrx_gain = 0
        self.samp_rate = samp_rate = 1E8/16
        self.samp_freq = samp_freq = 915E6
        self.rx_antenna = rx_antenna = "TX/RX"
        self.output_file = output_file = "cal_rx_step915.raw"

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
        	",".join(("addr=192.168.2.203", "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.uhd_usrp_source_0.set_clock_source("external", 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(samp_freq, 0)
        self.uhd_usrp_source_0.set_gain(txrx_gain, 0)
        self.uhd_usrp_source_0.set_antenna(rx_antenna, 0)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, output_file, False)
        self.blocks_file_sink_0.set_unbuffered(False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_file_sink_0, 0))    

    def get_txrx_gain(self):
        return self.txrx_gain

    def set_txrx_gain(self, txrx_gain):
        self.txrx_gain = txrx_gain
        self.uhd_usrp_source_0.set_gain(self.txrx_gain, 0)
        	

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_samp_freq(self):
        return self.samp_freq

    def set_samp_freq(self, samp_freq):
        self.samp_freq = samp_freq
        self.uhd_usrp_source_0.set_center_freq(self.samp_freq, 0)

    def get_rx_antenna(self):
        return self.rx_antenna

    def set_rx_antenna(self, rx_antenna):
        self.rx_antenna = rx_antenna
        self.uhd_usrp_source_0.set_antenna(self.rx_antenna, 0)

    def get_output_file(self):
        return self.output_file

    def set_output_file(self, output_file):
        self.output_file = output_file
        self.blocks_file_sink_0.open(self.output_file)


def main(top_block_cls=cal_rx, options=None):

    raw_input("Connect 10nW(-50dBm) CW calibration source at 2,451,625.000 Hz and press Enter to continue...")

    tb = top_block_cls()

    tb.start()
    mygain = 0
    for i in range(0,32):
        tb.uhd_usrp_source_0.set_gain(i, 0)
        time.sleep(0.5)
    
    tb.stop()
    tb.wait()

    raw_input("Turn off calibration source and terminate input with termination cap and press Enter to continue...")

    tb.start()
    mygain = 0
    for i in range(0,32):
        tb.uhd_usrp_source_0.set_gain(i, 0)
        time.sleep(0.5)
    
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
