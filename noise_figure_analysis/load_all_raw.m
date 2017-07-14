%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Signal Laboratories, Inc.
% (c) 2016. Joel D. Brinton.
%
% Load .RAW file (from GNU Radio Companion)
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = load_raw(file_name)

f = dir(file_name);
len = f.bytes / 4;

fid = fopen(file_name, 'r');

b = fread(fid, len, 'single');

fclose(fid);

out = complex(b(1:2:end), b(2:2:end));

