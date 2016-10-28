% Copyright (C) 2016 Ameya Patil
% 
% This program is free software; you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% -*- texinfo -*- 
% @deftypefn {Function File} {@var{retval} =} cal_proceedure (@var{input1}, @var{input2})
%
% @seealso{}
% @end deftypefn

% Author: Ameya Patil <ameya@ameya>
% Created: 2016-10-13

function [gain_error_y_int_db, psd_n_cal_dbmhz] = cal_procedure (cal_recording, rx_dsf, tx_dsf, tone_freq)

%
% This function is expecting a time series as follows:
%
% @@@@%%%%%%%%%%%%%%
%
% where @ is a recording of a 1uW CW tone and
% # is a recording of just noise (resistor terminated receiver)
%
% The first @@@@ period should be 1.000 seconds
% Each subsequent period should be 1.000 seconds and have gain_steps gain
%
% rx_dsf : RX down sample factor compared to max SDR rate (6.25e6)
% tx_dsf : TX down sample factor compared to max SDR rate (6.25e6)

% load Siglabs Utilities
%o_util;

% cal file sample rate
%sps = 6.25e6/4;
sps = 6.25e6/rx_dsf;

% samples per test sequence
%spts = 3.125e6/4;
spts = 3.125e6/rx_dsf;

% cal single tone power (dBm)
cstp_dbm = -50;

% start of time series zero pading index (to remove glitches)
% begin time series at 100ms
stszpi = sps*0.100 + 1;

% end of time series zero padding index (to remove glitches)
% end time series at 400ms
etszpi = sps*0.400;

% calibration channel start (Hz)
%ccs = 1620e3/4/8;
%ccs = 1620e3/tx_dsf;
ccs = tone_freq - 10e3;

% calibration channel end (Hz)
%cce = 1630e3/4/8;
%cce = 1630e3/tx_dsf;
cce = tone_freq + 10e3;

% noise channel start (Hz)
%ncs = 625e3/4;
ncs = 625e3/rx_dsf;

% noise channel end (Hz)
%nce = 2500e3/4;
nce = 2500e3/rx_dsf;

% gain steps dB (as recorded)
sdr_gain_steps = 0:31;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% OPEN FILE AND PARSE
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% recording of 1uW single tone at 1 MHz with SDR gain = 0 (0 to 31.5)
%cal_recording = rawfile_to_complex(cal_file);

% separate calibration tone recordings (first 32 data sets)
gain_recording = cal_recording(1:spts*32);
gain_recording = reshape(gain_recording, spts, 32);

% chop 100ms off ends to remove switching and startup transients
gain_recording = gain_recording(stszpi:etszpi, :);

% separate noise floor recordings
% (subsequent 32 seconds, less 1000 samples to avoid EOF issues)
noise_recording = cal_recording(spts*32+1:spts*64-1000);
noise_recording(end+1:end+1000) = 0;
noise_recording = reshape(noise_recording, spts, 32);

% chop off ends to remove switching and startup transients
noise_recording = noise_recording(stszpi:etszpi,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CALCULATE GAIN ERROR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i = 1:32;
   [out, powerW, rmsV, psd_dbWhz] = cal_filter(gain_recording(:,i),sps,ccs,cce);

    % band power
    power_dbm = 10 * log10(powerW) + 30;

    meas_gain_error_db(i) = power_dbm - cstp_dbm;
end

gain_error_y_int_db = meas_gain_error_db(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CALCULATE NOISE FLOOR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:32;
   [out, powerW, rmsV, psd_dbWhz] = cal_filter(noise_recording(:,i),sps,ncs,nce);

    % band power spectral density
    psd_n_cal_dbWhz(i) = psd_dbWhz - (gain_error_y_int_db + sdr_gain_steps(i));
    %psd_n_cal_dbWhz(i) = psd_dbWhz - meas_gain_error_db(i);
    psd_n_cal_dbmhz(i) = psd_n_cal_dbWhz(i) + 30;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CALCULATE FULL SCALE INPUT
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate dummy "full scale" input signal
fs_recording = ones(spts,1);
fs_recording = fs_recording(stszpi:etszpi,:);

for i = 1:32;
   [out, powerW, rmsV, psd_dbWhz] = cal_filter(fs_recording,sps,-1000,1000);

   % band power
   power_dbW = 10 * log10(powerW);
   power_dbm = power_dbW + 30;
   
   fs_cal_dbm(i) = power_dbm - meas_gain_error_db(i);
end


figure;

subplot(2,2,1);
temp1 = cal_recording(1:100:end);
temp1 = abs(temp1);
templ = size(temp1, 1);
tempp = 100/sps;
tempx = (tempp:tempp:(tempp*templ))';
plot(tempx, temp1);
title('Siglabs Suitcase Receiver S/N 001');
xlabel('time (s)');
ylabel('calibration signal sequence (Volts)');

subplot(2,2,2);
plot(sdr_gain_steps, psd_n_cal_dbmhz);
title('Siglabs Suitcase Receiver S/N 001');
xlabel('SDR gain value (db)');
ylabel('Noise Power Spectral Density (dBm/Hz)');

subplot(2,2,3);
plot(sdr_gain_steps, fs_cal_dbm);
title('Siglabs Suitcase Receiver S/N 001');
xlabel('SDR gain value (db)');
ylabel('Full Scale Single Tone Input (dBm)');

subplot(2,2,4);
expected_gain_error = sdr_gain_steps + gain_error_y_int_db;
plot(sdr_gain_steps, [meas_gain_error_db; expected_gain_error]);
title('Siglabs Suitcase Receiver S/N 001');
xlabel('SDR gain value (db)');
ylabel('Measured Gain Error (dB)');

