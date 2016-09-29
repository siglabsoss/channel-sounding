all: drive_rx.py

drive_rx.py: drive-rx.grc
	grcc drive-rx.grc -d .

clean:
	rm -f drive_rx.py drive_tx.py top_block.py

