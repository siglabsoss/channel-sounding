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

function out = load_raw(file_name, calibration_factor, sps, start, stop)

f = dir(file_name);
len = f.bytes / 4;

fid = fopen(file_name, 'r');

start_s = start * sps * 2;

if( start_s > len )
    error('start begins after end of file!');
end

stop_s = stop * sps * 2;

if( stop_s > len )
    disp(sprintf('stop is after end of file... new stop = %d seconds', len / (sps * 2)));
    
    stop_s = len;
end

fseek(fid, start_s * 4, 0);

b = fread(fid, stop_s - start_s, 'single');

fclose(fid);

out = complex(b(1:2:end), b(2:2:end)) ./ calibration_factor;

