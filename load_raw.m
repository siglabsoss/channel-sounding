%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Signal Laboratories, Inc.
% (c) 2016. Joel D. Brinton.
%
% Load .RAW file (from GNU Radio Companion)
%
%
% Notes: This Calibration Factor is the linear gain error in units
%        of Volts (in a 1-Ohm system).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = load_raw(file_name, calibration_factor)

f = dir(file_name);
len = f.bytes;

fid = fopen(file_name, 'r');

b = fread(fid, len/4, 'single');

out = complex(b(1:2:end), b(2:2:end)) ./ calibration_factor;

