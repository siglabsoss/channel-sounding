clear

%%%%%%%%%%%%%%%%%%%%%%%
% parameters
%
% spectral noise density always in dBm/Hz

% 60k samples
adc_nf_start = 1233000;
adc_nf_stop = 1293000;

rf_nf_start = 1616000;
rf_nf_stop = 1676000;

adc_fs = 10; % dBm
adc_snr = 73.2; % dB
sps = 250e6;

rf_afe_gain = 52.55; % dB (calculated)

johnson_noise_db = -174; % dBm/Hz


%%%%%%%%%%%%%%%%%%%%%%%
% load raw ADC data

load 170817-1_adc_samples_high_gain.mat

x = adcsampleshighgain;

%%%%%%%%%%%%%%%%%%%%%%%
% calculate noise floor

% ADC nose floor (datasheet)
adc_noise_ds_db = adc_fs - adc_snr - 10 * log10(sps);

% RF spectrum (double sided)
xf = fft(x);

% measure noise levels (relative)
adc_noise = mean(abs(xf(adc_nf_start:adc_nf_stop)).^2);
rf_noise = mean(abs(xf(rf_nf_start:rf_nf_stop)).^2);

adc_noise_db = 10*log10(adc_noise);
rf_noise_db = 10*log10(rf_noise);

% ADC calibration
adc_cal = adc_noise_db - adc_noise_ds_db;

% calibrate
adc_noise_db = adc_noise_db - adc_cal;
rf_noise_db = rf_noise_db - adc_cal;

meas_rf_noise_db = rf_noise_db - rf_afe_gain

rf_noise_figure = meas_rf_noise_db - johnson_noise_db



