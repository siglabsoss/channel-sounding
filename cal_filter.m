function [out, power, rms, psd_dbhz] = cal_filter(in, sps)

% size of data set
l = size(in,1);

% period of data set
p = l / sps;

% NOTE: we're only going to use the last half of the time series
% because the beginning usually gets corrupted by the USRP filters

% band pass filter start
bpfs = round(0.1 * l / 2);

% band pass filter end
bpfe = round(0.4 * l / 2);

% compute FT of last half of the time series
inf = fft(in(end/2:end));

% notch filter zero to BPFS
inf(1:bpfs) = 0;

% notch filter BPFE to zero
inf(bpfe:end) = 0;

% compute IFT of filtered values
inif = ifft(inf);

% compute RMS for band and compensate for notch filter (30% power)
rms = sqrt(mean( (abs(inif) .^ 2) / 0.3 ));

% compute average power
power = rms^2 / 50;

% compute PSD
psd_dbhz = 10 * log10(power / sps);


out = inif;