function [out, powerW, rmsV, psd_dbWhz, inft] = cal_filter(in, sps, flt_start, flt_stop)

% size of data set
l = size(in,1);

% resolution bandwidth
rbw = sps / l;

% NOTE: we're only going to use the last half of the time series
% because the beginning usually gets corrupted by the USRP filters

% band pass filter start bin
bpfs = round(flt_start/rbw + (l/2) + 1);

% band pass filter end bin
bpfe = round(flt_stop/rbw + (l/2));

% fractional bandwidth
bw_f = (bpfs - bpfe) / l;

% compute FT of last half of the time series
inft = fftshift(fft(in));

% zero pad
inft(1:bpfs) = 0;
inft(bpfe:end) = 0;

% brick wall filter and compute IFT
inift = ifft(ifftshift(inft));

% compute RMS for band
rmsV = rms(inift);

% compute average BPF per sample power on a 1-ohm transmission line
powerW = rmsV.^2 / 1;

% compute PSD
bw = flt_stop - flt_start;
psd_dbWhz = 10 * log10(powerW/bw);


out = inift;