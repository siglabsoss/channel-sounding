#!/bin/bash

sudo ifdown enx74da38700fc4
sudo ifup enx74da38700fc4
sleep 5
sudo service openvpn restart


