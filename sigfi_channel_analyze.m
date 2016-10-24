%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Signal Laboratories, Inc.
% (c) 2016. Joel D. Brinton.
%
% RF Propagation Sounding Analysis
%
%
% Notes: This function assumes that there is zero Sample Frequency Offset
%        for the entirety of the sample. At 6.25 Msps and 60 seconds of
%        sample data this means that the sampling oscillator must be much
%        less than 2.7ppb (ideally less than 1.0ppb). Using Rubidium
%        Standards this is possible.
%
%        This function assumes that there is negligible frequency drift
%        for the entirety of the sample. A single Channel Frequency
%        Offset (CFO) is computed for the entire run.
%
%        This function assumes that the PN sequence was transmitted at
%        50% duty cycle.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function multipath_vec = sigfi_channel_analyze(X, fs, ref, fs_ref, tx_dbm, txant_db, rxant_db)

% sample rate to filter sample rate ratio
% round because floating point error will cause indexing error
% Note: the sample rate to filter sample rate ratio needs to be integer!
srfsr = round(fs / fs_ref);

% up-sample factor
% Note: if the signal is not oversampled then do additional oversampling
usf = ceil(5 / srfsr);

% sample rate
ts = 1/fs;
fs_int = fs * usf;
ts_int = ts / usf;
X_len = length(X);

% duty cycle of PN sequence
dc = 0.5;

% doppler averaging period (Hz)
% Note: THIS IS WHERE YOU SET HOW LONG YOU AVERAGE EACH DOPPER POINT
%        10Hz = 0.1 seconds
%         2Hz = 0.5 seconds
%       0.5Hz = 2 seconds
%       0.1Hz = 10 seconds
%        etc....
dplr = 0.5;

% process gain
pg = length(ref) * srfsr;
pg_int = pg * usf;

% sample length
X_l = length(X);

% sample period
X_p = X_l / fs;

% nearby interferer auto-gain threshold
% Note: THIS IS THE THRESHOLD IN VOLTS @ 1-OHM (SCALED) WHEN THE CHANNEL
%       IS MUTED DUE TO NEARBY IN-BAND INTERFERENCE.
%       NEED TO ADJUST THIS TO MAXIMIZE CORRELATION PERFORMANCE.
thresh = 1e-8;


%%%%%%%%%%%%%%%%%%%%%%%
% UP-SAMPLE AND NORMALIZE MATCHED FILTER
%%%%%%%%%%%%%%%%%%%%%%%

% up-sample to sample rate
ref = interp(double(ref), srfsr);

% up-sample matched filter to remove Sample Phase Offset
ref_int = interp(double(ref),usf);

% normalize match filter
ref_int_norm = ref_int./sqrt(ref_int'*ref_int);


%%%%%%%%%%%%%%%%%%%%%%%
% CORRECT CFO
%%%%%%%%%%%%%%%%%%%%%%%

% time series
t = ts:ts:X_p;

% TODO: measure CFO

% CFO in radians
cfo = 0 * 2 * pi;

% local oscillator
lo = exp(1j*cfo*t);

% mix
X = X .* lo';

% TODO: CORRECT FOR RUBIDIUM BEATING

%%%%%%%%%%%%%%%%%%%%%%%
% FILTER
%%%%%%%%%%%%%%%%%%%%%%%

[X_filt, ~, ~, ~, ~] = cal_filter(X, fs, -fs/(2*srfsr), fs/(2*srfsr));

%%%%%%%%%%%%%%%%%%%%%%%
% AUTO-GAIN TO REJECT NEARBY IN-BAND INTERFERERS
%%%%%%%%%%%%%%%%%%%%%%%

X_filt_n_idx = find(abs(X_filt) > thresh);
X_filt(X_filt_n_idx) = 0;

% this interpolates data when theres missing samples due to nearby
% in-band interferers
%X_filt_int(1:(pg_int/dc)) = 0;
%X_filt_int_n_idx = find(abs(X_filt_int(pg_int/dc+1:end)) > thresh);
%X_filt_int(X_filt_int_n_idx + (pg_int/dc)) = X_filt_int(X_filt_int_n_idx);


%%%%%%%%%%%%%%%%%%%%%%%
% AVERAGE SIGNAL
%%%%%%%%%%%%%%%%%%%%%%%

% PN sequence period
period = pg / dc;

% number of PN sequence periods in recording
n_max = floor(length(X_filt) / period);

% number of PN sequences in averaging window
n_pav = floor(fs / period / dplr);

% number of averaging windows in recording
n_awr = floor(n_max / n_pav);

if(n_awr < 2)
    error('Need at least 2 Doppler periods to measure channel');
end

X_ave_cp(period * 2, n_awr) = 0;

for idx = 1:n_awr

    % chop recording up into equal averaging periods
    X_ave = reshape(X_filt(1:n_max*period), period, n_max);

    % average
    X_s = (idx-1)*n_pav+1;
    X_e = idx * n_pav;
    X_ave = sum(X_ave(:,X_s:X_e), 2)./(n_pav);

    % add cyclic prefix and sufix
    X_ave_cp(:,idx) = [X_ave(end/2+1:end); X_ave; X_ave(1:end/2)];

end


%%%%%%%%%%%%%%%%%%%%%%%
% INTERPOLATE
%%%%%%%%%%%%%%%%%%%%%%%

X_filt_int(period * 2 * usf, n_awr) = 0;

for idx = 1:n_awr
    X_filt_int(:,idx) = interp(double(X_ave_cp(:,idx)), usf);
end


%%%%%%%%%%%%%%%%%%%%%%%
% CROSS CORRELATION
%%%%%%%%%%%%%%%%%%%%%%%

% preallocate array
X_xcr(pg_int/dc,n_awr) = 0;

for idx = 1:n_awr
    
    % cross-correlaton
    X_xcr_p = xcorr(X_filt_int(:,idx), ref_int_norm);

    % remove cyclic prefix and sufix
    X_xcr(:,idx) = X_xcr_p(2*pg_int/dc+1:3*pg_int/dc);
    
end

%%%%%%%%%%%%%%%%%%%%%%%
% AVERAGE MULTIPLE CROSS CORRELATIONS
%%%%%%%%%%%%%%%%%%%%%%%

X_xcr_ave = sum(abs(X_xcr),2) ./ size(X_xcr,2);


%%%%%%%%%%%%%%%%%%%%%%%
% MULTI-PATH PEAK-TO-AVERAGE RATIO (PAR) 
%%%%%%%%%%%%%%%%%%%%%%%

% find peak
[~,idx] = max(X_xcr_ave);

% calculate peak to average (PAR) ratio)
X_xcr_ave_par = max(X_xcr_ave) / mean(X_xcr_ave);

% if there's no strong multi-path peak then there's nothing you can do!!!
if (X_xcr_ave_par < 3)
    error('No sufficiently strong multi-paths'); 
end

%%%%%%%%%%%%%%%%%%%%%%%
% CIRCLE-SHIFT
% Note: this puts peak in center of plot
%%%%%%%%%%%%%%%%%%%%%%%

cs_len = size(X_xcr_ave,1);
cs_mid = floor(cs_len / 2);

X_xcr_ave = circshift(X_xcr_ave, cs_mid - idx);
X_xcr = circshift(X_xcr, cs_mid - idx, 1);


%%%%%%%%%%%%%%%%%%%%%%%
% MEASURE CFO
% Note: this doesn't work well at low SNR.
% Note: decrease averaging window to get better measurements.
%%%%%%%%%%%%%%%%%%%%%%%

% measure change in phase angle of largest multi-path
xcr_tap = X_xcr(cs_mid,:);
xcr_tap_ang = unwrap(angle(xcr_tap));
xcr_tap_ang_diff = diff(xcr_tap_ang);

% generate y-axis (for plots)
tcfo_int = n_pav * period * ts;
tcfo_mp = tcfo_int * length(xcr_tap_ang_diff);
tcfo = (tcfo_int):tcfo_int:tcfo_mp;

% scale to Hertz
xcr_tap_ang_diff = xcr_tap_ang_diff / (2 * pi * tcfo_int);

%%%%%%%%%%%%%%%%%%%%%%%
% MULTI-PATH STATISTICS
%%%%%%%%%%%%%%%%%%%%%%%

% capture 10us before and after
%spread = floor (10e-6 / ts_int);

% capture 256 samples before and after peak
spread = 256;

% window multipath vectors to region of interest (i.e. center +/- spread)
multipath_vec = X_xcr(cs_mid-spread+1:cs_mid+spread,:);

% generate x-axis (for plots)
t_mp = length(multipath_vec(:,1)) / (fs * usf);
t = ts_int:ts_int:t_mp;
t2d = t' * ones(1,length(multipath_vec(1,:)));

% generate y-axis (for plots)
y_mp = length(multipath_vec(1,:));
y = (1/dplr):(1/dplr):(y_mp/dplr);
y2d = ones(length(multipath_vec(:,1)),1) * y;

% SUM ALL MULTIPATH AVERAGING WINDOWS TOGETHER (i.e. ENTIRE RECORDING)
% Note: THIS IS NOT YET SCALED PROPERLY.. PROBABLY NEED TO DIVIDE
%       BY NUMBER OF AVERAGING WINDOWS THAT ARE BEING SUMMED???
multipath_ave = X_xcr_ave(cs_mid-spread+1:cs_mid+spread,:);

% TODO: PASS BACK MULTI-PATH STATISTICS TO CALLING FUNCTION SO THAT
%       THEY MAY BE TABULATED


%%%%%%%%%%%%%%%%%%%%%%%
% MEASURE DOPPLER POWER SPECTRUM
%%%%%%%%%%%%%%%%%%%%%%%

dfs = -dplr+(dplr/size(multipath_vec,2)*2):dplr/size(multipath_vec,2)*2:dplr;
dps = abs(fftshift(fft(multipath_vec,[],2),2))';

% normalize power
dps = dps ./ max(max(abs(dps)));

% TODO: PASS BACK DOPPLER STATISTICS TO CALLING FUNCTION SO THAT THEY
%       MAY BE TABULATED

%%%%%%%%%%%%%%%%%%%%%%%
% EQUALIZE
%%%%%%%%%%%%%%%%%%%%%%%

% TODO

%%%%%%%%%%%%%%%%%%%%%%%
% MEASURE POWER
%%%%%%%%%%%%%%%%%%%%%%%

% TODO

%%%%%%%%%%%%%%%%%%%%%%%
% MEASURE PATH LOSS
%%%%%%%%%%%%%%%%%%%%%%%

% TODO

% TODO: PASS BACK PATH LOSS TO CALLING FUNCTION SO THAT STATISTICS CAN
%       BE MEASURED.


%%%%%%%%%%%%%%%%%%%%%%%
% PLOTS PLOTS PLOTS
%%%%%%%%%%%%%%%%%%%%%%%

figure;
plot(dfs, dps);
xlabel('frequency (Hz)');
ylabel('normalized power');
title('Dopper Power Spectrum');

figure;
surf(dps,'EdgeColor','none');
set(gca,'ZScale','log');
view(0, 90);
title('Dopper Power Spectrum');

figure;
plot(tcfo, xcr_tap_ang_diff);
ylim([-1 1]);
xlabel('time (s)');
ylabel('frequency (Hz)');
title('Channel Frequency Offset');

figure;
ta = ts*(1000+1):ts*1000:X_len*ts;
xa = X_filt(1000+1:1000:end);
plot(ta,abs(xa));
xlabel('time (s)');
ylabel('amplitude (Volts @ 1-Ohm)');
title('test recording');

figure;
plot(t, multipath_ave);
hline(2e-10,'r:','threshold');
title('Average Multipath');
xlabel('time (s)');

figure;
plot(t, abs(multipath_vec));
xlabel('time (s)');
ylabel('power (W)');
title('Delay Spreads');

figure;
surf(t2d,y2d,abs(multipath_vec),'EdgeColor','none');
view(0, 90);
xlabel('spread (s)');
ylabel('time (s)');
zlabel('power (W)');
title('Delay Spreads');

autoArrangeFigures;






