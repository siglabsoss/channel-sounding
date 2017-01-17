all : drive_rx.py drive_tx.py  _rx_905E6.py  _rx_910E6.py  _rx_915E6.py  _rx_2410E6.py  _rx_2420E6.py  _rx_2430E6.py  _tx_905E6.py  _tx_910E6.py  _tx_915E6.py  _tx_2410E6.py  _tx_2420E6.py  _tx_2430E6.py 

drive_rx.py: drive-rx.grc
	grcc drive-rx.grc -d .

drive_tx.py: drive-tx.grc
	grcc drive-tx.grc -d .

clean:
	rm -f drive_rx.py drive_tx.py top_block.py *orig *pyc _*

_rx_905E6.py: drive_rx.py _premake_run
	cp drive_rx.py _rx_905E6.py
	patch _rx_905E6.py < _patch_905E6.patch
	patch _rx_905E6.py < sleep4.patch
	patch _rx_905E6.py < _patch_output_one_905E6.patch
	patch _rx_905E6.py < _patch_gain_3.patch


_rx_910E6.py: drive_rx.py _premake_run
	cp drive_rx.py _rx_910E6.py
	patch _rx_910E6.py < _patch_910E6.patch
	patch _rx_910E6.py < sleep4.patch
	patch _rx_910E6.py < _patch_output_one_910E6.patch
	patch _rx_910E6.py < _patch_gain_3.patch


_rx_915E6.py: drive_rx.py _premake_run
	cp drive_rx.py _rx_915E6.py
	patch _rx_915E6.py < _patch_915E6.patch
	patch _rx_915E6.py < sleep4.patch
	patch _rx_915E6.py < _patch_output_one_915E6.patch
	patch _rx_915E6.py < _patch_gain_3.patch


_rx_2410E6.py: drive_rx.py _premake_run
	cp drive_rx.py _rx_2410E6.py
	patch _rx_2410E6.py < _patch_2410E6.patch
	patch _rx_2410E6.py < sleep4.patch
	patch _rx_2410E6.py < _patch_output_one_2410E6.patch
	patch _rx_2410E6.py < _patch_gain_10.patch
	patch _rx_2410E6.py < change_antenna.patch


_rx_2420E6.py: drive_rx.py _premake_run
	cp drive_rx.py _rx_2420E6.py
	patch _rx_2420E6.py < _patch_2420E6.patch
	patch _rx_2420E6.py < sleep4.patch
	patch _rx_2420E6.py < _patch_output_one_2420E6.patch
	patch _rx_2420E6.py < _patch_gain_10.patch
	patch _rx_2420E6.py < change_antenna.patch


_rx_2430E6.py: drive_rx.py _premake_run
	cp drive_rx.py _rx_2430E6.py
	patch _rx_2430E6.py < _patch_2430E6.patch
	patch _rx_2430E6.py < sleep4.patch
	patch _rx_2430E6.py < _patch_output_one_2430E6.patch
	patch _rx_2430E6.py < _patch_gain_10.patch
	patch _rx_2430E6.py < change_antenna.patch


_tx_905E6.py: drive_tx.py _premake_run
	cp drive_tx.py _tx_905E6.py
	patch _tx_905E6.py < _patch_905E6.patch
	patch _tx_905E6.py < _patch_gain_5.patch
	patch _tx_905E6.py < sleep2.patch


_tx_910E6.py: drive_tx.py _premake_run
	cp drive_tx.py _tx_910E6.py
	patch _tx_910E6.py < _patch_910E6.patch
	patch _tx_910E6.py < _patch_gain_5.patch
	patch _tx_910E6.py < sleep2.patch


_tx_915E6.py: drive_tx.py _premake_run
	cp drive_tx.py _tx_915E6.py
	patch _tx_915E6.py < _patch_915E6.patch
	patch _tx_915E6.py < _patch_gain_5.patch
	patch _tx_915E6.py < sleep2.patch


_tx_2410E6.py: drive_tx.py _premake_run
	cp drive_tx.py _tx_2410E6.py
	patch _tx_2410E6.py < _patch_2410E6.patch
	patch _tx_2410E6.py < _patch_gain_24.patch
	patch _tx_2410E6.py < sleep2.patch


_tx_2420E6.py: drive_tx.py _premake_run
	cp drive_tx.py _tx_2420E6.py
	patch _tx_2420E6.py < _patch_2420E6.patch
	patch _tx_2420E6.py < _patch_gain_24.patch
	patch _tx_2420E6.py < sleep2.patch


_tx_2430E6.py: drive_tx.py _premake_run
	cp drive_tx.py _tx_2430E6.py
	patch _tx_2430E6.py < _patch_2430E6.patch
	patch _tx_2430E6.py < _patch_gain_24.patch
	patch _tx_2430E6.py < sleep2.patch


runrx:  _rx_905E6.py  _rx_910E6.py  _rx_915E6.py  _rx_2410E6.py  _rx_2420E6.py  _rx_2430E6.py 
	sudo ls > /dev/null
	sudo python ./runner.py type rx
	@sudo touch /mnt/usb1/test_one/"`date`"
	@echo made these files
	@echo 
	ls -lsh /mnt/usb1/test_one


runtx:  _tx_905E6.py  _tx_910E6.py  _tx_915E6.py  _tx_2410E6.py  _tx_2420E6.py  _tx_2430E6.py 
	sudo ls > /dev/null
	sudo python ./runner.py type tx



.PHONY: runrx runtx


