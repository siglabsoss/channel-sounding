%%%%
% Single-tone Noise Figure Analaysis
%
% 1. connect CALIBRATED REFERENCE TONE to receiver
% 2. transmit tone followed by silence
% 3. scale waveform to match CALIBRATED REFERENCE TONE power
% 4. measure band power during silence
%

%clear
close

% load coefficients for a 12/99.84 kHz filter
%load 170718-test-results.mat

sps = 99.84e3; % samples per second
noise_floor = -174; % dbm/hz
bw = sps / 8; % channel bandwidth (Hertz)
noise_power = -174 + 10 * log10(bw); % room temperature noise power

%%%%
% Premixer Gain = 32dB
% Postmixer Gain = 32dB
%
test1_rf_in = -51; % CALIBRATED REFERENCE TONE power (dBm)

test1_flt = filter(lpf,1,test1);

test1_on = test1_flt(1*sps+1:2*sps);

test1_off = test1_flt(6*sps+1:7*sps);

test1_on_dbm = 10*log10(sum(abs(test1_on).^2)) + 30; % dBm
test1_on_gain_meas_db = test1_on_dbm - test1_rf_in;

test1_off_dbm = 10*log10(sum(abs(test1_off).^2)) + 30; % dBm
test1_nf = test1_off_dbm - test1_on_gain_meas_db - noise_power;
fprintf('Noise Figure = %2.1fdB (programmable gain = 64dB), expected 1.1dB\n',...
    test1_nf);

semilogy(abs(fftshift(fft(test1_on))))
hold on
semilogy(abs(fftshift(fft(test1_off))))

%%%%
% Premixer Gain = 0dB
% Postmixer Gain = 0dB
%
test2_rf_in = -18.3; % CALIBRATED REFERENCE TONE power (dBm)

test2_flt = filter(lpf,1,test2);

test2_on = test2_flt(1*sps+1:2*sps);

test2_off = test2_flt(6*sps+1:7*sps);

test2_on_dbm = 10*log10(sum(abs(test2_on).^2)) + 30; % dBm
test2_on_gain_meas_db = test2_on_dbm - test2_rf_in;

test2_off_dbm = 10*log10(sum(abs(test2_off).^2)) + 30; % dBm
test2_nf = test2_off_dbm - test2_on_gain_meas_db - noise_power;
fprintf('Noise Figure = %2.1fdB (programmable gain = 0dB), expected 45.8dB\n',...
    test2_nf);



