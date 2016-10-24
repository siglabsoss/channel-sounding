#!/bin/bash

sudo ifdown enx74da382af31f
sudo ifup enx74da382af31f
sleep 5
sudo service openvpn restart


