all: drive_rx915.py drive_rx24.py drive_tx915.py drive_tx24.py

drive_rx915.py: drive-rx.grc
	grcc drive-rx.grc -d .
	mv drive_rx.py drive_rx915.py
	patch drive_rx915.py < sleep.patch

drive_rx24.py: drive-rx.grc
	grcc drive-rx.grc -d .
	mv drive_rx.py drive_rx24.py
	patch drive_rx24.py < sleep.patch
	patch drive_rx24.py < changehz.patch
	patch drive_rx24.py < changeantenna.patch
	

drive_tx915.py: drive-tx.grc
	grcc drive-tx.grc -d .
	mv drive_tx.py drive_tx915.py
	patch drive_tx915.py < sleep1.patch

drive_tx24.py: drive-tx.grc
	grcc drive-tx.grc -d .
	mv drive_tx.py drive_tx24.py
	patch drive_tx24.py < sleep1.patch
	patch drive_tx24.py < changehz.patch


runrx915: drive_rx915.py
	rm -f /mnt/usb1/file1.raw
	sudo ./drive_rx915.py

runrx24: drive_rx24.py
	rm -rf /mnt/usb1/file1.raw
	sudo ./drive_rx24.py

runtx915: drive_tx915.py
	sleep 1
	sudo ./drive_tx915.py
	sleep 1

runtx24: drive_tx24.py
	sleep 1
	sudo ./drive_tx24.py
	sleep 1


clean:
	rm -f drive_rx.py drive_tx915.py drive_tx24.py top_block.py drive_rx915.py drive_rx24.py *orig

