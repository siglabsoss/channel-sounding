clear
close

load lowgain.mat
load highgain.mat

sps = 99.84e3;
lge = -20.4; % calibrated input power (low gain test)
hge = -90; % calibrated input power (high gain test)
noise_floor = -174; % dbm/hz
noise_power = -174 + 10 * log10(sps);

low_gain_loud = lowgain(1:1*sps);
%plot(real(low_gain_loud))

low_gain_quiet = lowgain(6*sps+1:7*sps);
plot(real(low_gain_quiet))

high_gain_loud = highgain(1:1*sps);
%plot(real(high_gain_loud))

figure

high_gain_quiet = highgain(5*sps+1:6*sps);
%plot(real(high_gain_quiet))

hgl = 10*log10(sum(abs(high_gain_loud).^2)/50) + 30;
hgq = 10*log10(sum(abs(high_gain_quiet).^2)/50) + 30;

hgnf = (hge - (hgl - hgq)) - noise_power

lgl = 10*log10(sum(abs(low_gain_loud).^2)/50) + 30;
lgq = 10*log10(sum(abs(low_gain_quiet).^2)/50) + 30;

lgnf = (lge - (lgl - lgq)) - noise_power
