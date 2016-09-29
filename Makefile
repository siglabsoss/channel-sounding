all: drive_rx.py drive_tx.py

drive_rx.py: drive-rx.grc
	grcc drive-rx.grc -d .
	patch drive_rx.py < sleep.patch

drive_tx.py: drive-tx.grc
	grcc drive-tx.grc -d .

runrx: drive_rx.py
	rm -f /mnt/usb1/file1.raw
	sudo ./drive_rx.py

clean:
	rm -f drive_rx.py drive_tx.py top_block.py

