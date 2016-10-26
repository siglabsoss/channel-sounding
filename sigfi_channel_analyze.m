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

function [P_loss_db, tau, sigma_tau] = sigfi_channel_analyze(X, title_str, fs, ref, fs_ref, tx_dbm, txant_db, rxant_db)

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
dplr = 2;

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
%thresh = 1.7e-6;
thresh = 3e-6;

% RMS delay spread threshold (dB)
rms_thresh = 10;


%%%%%%%%%%%%%%%%%%%%%%%
% UP-SAMPLE AND NORMALIZE MATCHED FILTER
%%%%%%%%%%%%%%%%%%%%%%%

% up-sample to sample rate
ref = interp(double(ref), srfsr);

% up-sample matched filter to remove Sample Phase Offset
ref_int = interp(double(ref),usf);

% normalize match filter
%ref_int_norm = ref_int./sqrt(ref_int'*ref_int);
ref_int_norm = ref_int./sum(abs(ref_int).^2);


%%%%%%%%%%%%%%%%%%%%%%%
% CORRECT CFO
%%%%%%%%%%%%%%%%%%%%%%%

% time series
t_crop = ts:ts:X_p;

% TODO: measure CFO

% CFO in radians
cfo = 0 * 2 * pi;

% local oscillator
lo = exp(1j*cfo*t_crop);

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
    % because PN sequences are coherent over averaging window
    % just take the mean
    X_ave = mean(X_ave(:,X_s:X_e), 2);

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
% Notes: See http://tinyurl.com/hfgyy72
%        section 2.3 Sliding Correlator Theory
%        section 2.4 Implementation of a Sliding Correlator Measurement System
%        section 2.5 Analysis of Impulse Response Data
%
%        This routine uses the Swept Time-Delay Cross-Correlation method
%%%%%%%%%%%%%%%%%%%%%%%

% preallocate array
P(pg_int/dc,n_awr) = 0;

for idx = 1:n_awr
    
    % cross-correlaton
    h_b_cp = xcorr(X_filt_int(:,idx), ref_int_norm);

    % remove cyclic prefix and sufix
    h_b = h_b_cp(2*pg_int/dc+1:3*pg_int/dc);
    
    % calculate Power Delay Profile [RAP96a, (4.15)]
    P(:,idx) = 1 * abs(h_b).^2;
    
    % channel response matrix
    h_b_vec(:,idx) = h_b;
    
end

%%%%%%%%%%%%%%%%%%%%%%%
% AVERAGE MULTIPLE POWER DELAY PROFILES
%%%%%%%%%%%%%%%%%%%%%%%

P_ave = mean(abs(P),2);


%%%%%%%%%%%%%%%%%%%%%%%
% POWER DELAY PROFILE PEAK-TO-AVERAGE RATIO (PAR)
%%%%%%%%%%%%%%%%%%%%%%%

% find peak
[~,idx] = max(P_ave);

% calculate peak to average (PAR) ratio)
X_xcr_ave_par = max(P_ave) / mean(P_ave);

% if there's no strong multi-path peak then there's nothing you can do!!!
if (X_xcr_ave_par < 10)
    error('No sufficiently strong multi-paths'); 
end

%%%%%%%%%%%%%%%%%%%%%%%
% CIRCLE-SHIFT
% Note: this puts peak in center of plot
%%%%%%%%%%%%%%%%%%%%%%%

cs_len = size(P_ave,1);
cs_mid = floor(cs_len / 2);

P_ave = circshift(P_ave, cs_mid - idx);
P = circshift(P, cs_mid - idx, 1);
h_b_vec = circshift(h_b_vec, cs_mid - idx, 1);


%%%%%%%%%%%%%%%%%%%%%%%
% MEASURE CFO
% Note: this doesn't work well at low SNR.
% Note: decrease averaging window to get better measurements.
%%%%%%%%%%%%%%%%%%%%%%%

% measure change in phase angle of largest multi-path
P_tap = P(cs_mid,:);
P_tap_ang = unwrap(angle(P_tap));
P_tap_ang_diff_rad = diff(P_tap_ang);

% generate y-axis (for plots)
tcfo_int = n_pav * period * ts;
tcfo_mp = tcfo_int * length(P_tap_ang_diff_rad);
tcfo = (tcfo_int):tcfo_int:tcfo_mp;

% scale to Hertz
P_tap_ang_diff_hz = P_tap_ang_diff_rad / (2 * pi * tcfo_int);

%%%%%%%%%%%%%%%%%%%%%%%
% MULTI-PATH STATISTICS
%%%%%%%%%%%%%%%%%%%%%%%

% capture 10us before and after
%spread = floor (10e-6 / ts_int);

% capture 256 samples before and after peak
spread = 256;

% window multipath vectors to region of interest (i.e. center +/- spread)
P_crop = P(cs_mid-spread+1:cs_mid+spread,:);
P_crop_db = 10*log10(P_crop);
P_crop_ave = P_ave(cs_mid-spread+1:cs_mid+spread);
P_crop_ave_db = 10*log10(P_crop_ave);

h_b_vec_crop = h_b_vec(cs_mid-spread+1:cs_mid+spread,:);

% noise threshold in power
m = max(P_crop_ave);
ds_thresh = m / (10^(rms_thresh/10));
ds_thresh_db = 10*log10(ds_thresh);

% threshhold vector
P_crop_ave_zidx = find(P_crop_ave < ds_thresh);
P_crop_ave_thresh = P_crop_ave;
P_crop_ave_thresh(P_crop_ave_zidx) = 0;
P_crop_ave_thresh_db = 10 * log10(P_crop_ave_thresh);

% find first multi-path
P_crop_ave_thresh_pidx = find(abs(P_crop_ave_thresh) > 0);
t_first_multipath = P_crop_ave_thresh_pidx(1) * ts_int;

t_crop = (ts_int:ts_int:length(P_crop_ave_thresh)*ts_int)'...
         - t_first_multipath;
t_crop_us = t_crop * 1e6;


tau = sum(P_crop_ave_thresh .* t_crop) / sum(P_crop_ave_thresh);
tau2 = sum(P_crop_ave_thresh .* (t_crop .^ 2)) / sum(P_crop_ave_thresh);
sigma_tau = sqrt(tau2 - tau^2);

% maximum excess delay
med = t_crop(P_crop_ave_thresh_pidx(end));

% coherence bandwidth (0.9 frequency correlation)
bc = 1/(50 * sigma_tau);




% generate x-axis (for plots)
t2d = t_crop * ones(1,length(P_crop(1,:)));
t2d_us = t2d * 1e6;

% generate y-axis (for plots)
y_mp = size(P_crop,2);
y = (1/dplr):(1/dplr):(y_mp/dplr);
y2d = ones(length(P_crop_ave),1) * y;



% TODO: PASS BACK MULTI-PATH STATISTICS TO CALLING FUNCTION SO THAT
%       THEY MAY BE TABULATED


%%%%%%%%%%%%%%%%%%%%%%%
% MEASURE DOPPLER POWER SPECTRUM
%%%%%%%%%%%%%%%%%%%%%%%

%P_d_crop = P_crop(:,:);
P_d_crop = h_b_vec_crop;

%dfs_y_mp = size(P_d_crop,1);
%dfs_y = (1/dplr):(1/dplr):(dfs_y_mp/dplr);
%dfs2d_y = dfs_y * ones(1,size(P_d_crop,2));

doppler_step = (dplr/size(P_d_crop,2)*2);
dfs = -dplr:doppler_step:dplr-doppler_step;
dfs2d_x = ones(size(P_d_crop,1),1) * dfs;
dps = abs(fftshift(fft(P_d_crop,[],2),2))';

% normalize power
dps = dps ./ max(max(abs(dps)));

dps_db = 10*log10(dps);


% TODO: PASS BACK DOPPLER STATISTICS TO CALLING FUNCTION SO THAT THEY
%       MAY BE TABULATED

%%%%%%%%%%%%%%%%%%%%%%%
% EQUALIZE
%%%%%%%%%%%%%%%%%%%%%%%

% TODO

%%%%%%%%%%%%%%%%%%%%%%%
% MEASURE POWER
%%%%%%%%%%%%%%%%%%%%%%%

% calculate receive power (above noise)
% divide by process gain and up-sample factor
P_total = sum(P_crop_ave_thresh - ds_thresh) / pg / usf;
% convert to decible Watts
P_total_dbW = 10 * log10(P_total);
% convert to decibel mW
P_total_dbm = P_total_dbW + 30;


%%%%%%%%%%%%%%%%%%%%%%%
% MEASURE PATH LOSS
%%%%%%%%%%%%%%%%%%%%%%%

% calculate path loss
P_loss_db = tx_dbm - P_total_dbm + rxant_db + txant_db;

% TODO: PASS BACK PATH LOSS TO CALLING FUNCTION SO THAT STATISTICS CAN
%       BE MEASURED.


%%%%%%%%%%%%%%%%%%%%%%%
% PLOTS PLOTS PLOTS
%%%%%%%%%%%%%%%%%%%%%%%

disp(sprintf(['Mean Excess Delay = %0.1f(ns)\n'...
              'RMS Delay Spread = %0.1f(ns)\n'...
              'Max Excess Delay < %0.0fdB = %0.1f(ns)\n'...
              'Max Coherence Bandwidth (>0.9 correlation) = %0.0f(Hz)\n'...
              'Total Receive Power = %0.1fdBm\n',...
              'Total Path Loss = %0.1fdB\n'],...
              tau*1e9, sigma_tau*1e9, rms_thresh,...
              med*1e9, bc, P_total_dbm, P_loss_db ));

figure;
plot(t_crop_us, [P_crop_ave_db P_crop_ave_thresh_db], '-o');
xlim([t_crop_us(1) t_crop_us(end)]);
xlabel('time (\mus)');
ylabel('relative signal level (dB)');
hline(ds_thresh_db,'r:',sprintf('threshold from peak -%0.0fdB', rms_thresh));
title({'Average Multipath Power Delay Profile',...
       sprintf('file name: [%s]', title_str)});

%figure;
%plot(dfs, dps_db, '-o');
%xlim([-dplr dplr]);
%xlabel('frequency (Hz)');
%ylabel('normalized power');
%title('Dopper Power Spectrum');

figure;
surf(dfs2d_x, t2d_us, dps_db','EdgeColor','none');
ylim([t2d_us(1) t2d_us(end)]);
ylabel('delay tap (\mus)');
xlabel('Doppler frequency (Hz)');
zlabel('normalized power (dB)');
%view(0, 90);
title('Dopper Spread');

figure;
plot(tcfo, P_tap_ang_diff_hz);
xlim([tcfo(1) tcfo(end)]);
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
xlim([ta(1) ta(end)]);
title({'Test Recording',...
       sprintf('file name: [%s]', title_str)});


figure;
plot(t_crop_us, P_crop_db);
xlim([t_crop_us(1) t_crop_us(end)]);
xlabel('time (\mus)');
ylabel('relative signal level (dB)');
title({'All Multipath Power Delay Profiles',...
       sprintf('file name: [%s]', title_str)});

figure;
surf(t2d_us,y2d,P_crop_db,'EdgeColor','none');
xlim([t2d_us(1) t2d_us(end)]);
ylim([y2d(1) y2d(end)]);
view(90, 90);
xlabel('spread (\mus)');
ylabel('time (s)');
zlabel('relative signal level (dB)');
title({'All Multipath Power Delay Profiles',...
       sprintf('file name: [%s]', title_str)});

autoArrangeFigures;






