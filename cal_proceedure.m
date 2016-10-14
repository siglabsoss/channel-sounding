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

function [power_n_cal_db] = cal_proceedure (tone_file, noise_file, gain)

% load Siglabs Utilities
o_util;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CALCULATE GAIN ERROR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% recording of 1uW single tone at 1 MHz with SDR gain = 0 (0 to 31.5)
tone_recording = rawfile_to_complex(tone_file);

% compute channel power
% note that we're using 30% channel bandwidth to compute
% the energy of the tone... this is a sufficient approximation
[out, power, rms, psd_db] = cal_filter(tone_recording,6.25e6);

% compute channel power in dBm
power_db = 10 * log10(power);

% the gain error is the difference between 1uW and the channel power
error = power_db - (-30);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CALCULATE NOISE FLOOR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% recording of noise with SDR gain = gain
noise_record = rawfile_to_complex(noise_file);

% compute channel power
% note that we're using 30% channel bandwidth
[out_n, power_n, rms_n, psd_n_db] = cal_filter(noise_record,6.25e6);

% compute calibrated PSD
psd_n_cal_db = psd_n_db - error;

disp(psd_n_cal_db);

endfunction
