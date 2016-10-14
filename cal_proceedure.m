## Copyright (C) 2016 Ameya Patil
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} cal_proceedure (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Ameya Patil <ameya@ameya>
## Created: 2016-10-13

#function [gain_error, psd_n_cal_db] = cal_proceedure (cal_file)

%
% This function is expecting a time series as follows:
%
% @@@@############################
%
% where @ is a recording of a 1uW CW tone and
% # is a recording of just noise (resistor terminated receiver)
%
% The first @@@@ period should be 1.000 seconds
% Each subsequent period should be 1.000 seconds and have gain_steps gain
%

% load Siglabs Utilities
o_util;

% cal file sample rate
sps = 6.25e6;

% samples per test sequence
spts = 3.125e6;

% cal single tone power (dBm)
cstp = -50;

% start of time series zero pading index (to remove glitches)
% begin time series at 100ms
stszpi = sps*0.100 + 1;

% end of time series zero padding index (to remove glitches)
% end time series at 900ms
etszpi = sps*0.400;

% calibration channel start (Hz)
ccs = 1620e3;

% calibration channel end (Hz)
cce = 1630e3;

% noise channel start (Hz)
ncs = 625e3;

% noise channel end (Hz)
nce = 2500e3;

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

% chop 100ms off ends to remove switching and startup transients
noise_recording = noise_recording(stszpi:etszpi,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CALCULATE GAIN ERROR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for i = 1:32;
   [out, power, rms, psd_db] = cal_filter(gain_recording(:,i),sps,ccs,cce);

    power_db = 10 * log10(power);

    meas_gain_error_db(i) = power_db - cstp;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CALCULATE NOISE FLOOR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:32;
   [out, power, rms, psd_db] = cal_filter(noise_recording(:,i),sps,ncs,nce);

    psd_n_cal_db(i) = psd_db - meas_gain_error_db(i);
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
   [out, power, rms, psd_db] = cal_filter(fs_recording,sps,-1000,1000);

   power_db = 10 * log10(power);
   
    fs_cal_db(i) = power_db - meas_gain_error_db(i);
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
plot(sdr_gain_steps, psd_n_cal_db);
title('Siglabs Suitcase Receiver S/N 001');
xlabel('SDR gain value (db)');
ylabel('Noise Power Spectral Density (dBm/Hz)');

subplot(2,2,3);
plot(sdr_gain_steps, fs_cal_db);
title('Siglabs Suitcase Receiver S/N 001');
xlabel('SDR gain value (db)');
ylabel('Full Scale Single Tone Input (dBm)');

subplot(2,2,4);
yint = meas_gain_error_db(1);
expected_gain_error = sdr_gain_steps + yint;
plot(sdr_gain_steps, [meas_gain_error_db; expected_gain_error]);
title('Siglabs Suitcase Receiver S/N 001');
xlabel('SDR gain value (db)');
ylabel('Measured Gain Error (dB)');


#endfunction
