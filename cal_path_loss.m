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

function cal_path_loss(X, ref, tx_dbm, txant_db, rxant_db)

% up-sample factor
usf = 10;

% sample rate
fs = 6.25e6;
ts = 1/fs;
fs_int = fs * usf;
ts_int = ts / usf;

% duty cycle
dc = 0.5;

% doppler (Hz)
dplr = 0.5;

% process gain
pg = length(ref);
pg_int = pg * usf;

% sample length
X_l = length(X);

% sample period
X_p = X_l / fs;

%%%%%%%%%%%%%%%%%%%%%%%
% UP-SAMPLE AND NORMALIZE MATCHED FILTER
%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%
% CANCEL INTERFERERS
%%%%%%%%%%%%%%%%%%%%%%%

% TODO

%%%%%%%%%%%%%%%%%%%%%%%
% AVERAGE SIGNAL
%%%%%%%%%%%%%%%%%%%%%%%

% PN sequence period
period = pg / dc;

% number of PN sequence periods in recording
n_max = floor(length(X) / period);

% number of PN sequences in Doppler period
n_dplr = floor(fs / period / dplr);

% number of Doppler periods in recording
n_dpp = floor(n_max / n_dplr);

if(n_dpp < 2)
    error('Need at least 2 Doppler periods to measure channel');
end

% preallocate array
X_xcr(pg_int/dc,n_dpp) = 0;

for idx = 1:n_dpp

    % chop recording up into equally PN sequence periods
    X_ave = reshape(X(1:n_max*period), period, n_max);

    % average
    X_s = (idx-1)*n_dplr+1;
    X_e = idx * n_dplr;
    X_ave = sum(X_ave(:,X_s:X_e), 2)./n_dplr;

    % up-sample signal to remove Sample Phase Offset
    X_ave_int = interp(double(X_ave),usf);

    % add cyclic prefix and sufix
    X_ave_int_cp = [X_ave_int(end/2:end); X_ave_int; X_ave_int(1:end/2)];

    %%%%%%%%%%%%%%%%%%%%%%%
    % CROSS CORRELATION
    %%%%%%%%%%%%%%%%%%%%%%%

    % cross-correlaton
    X_xcr_p = xcorr(X_ave_int_cp, ref_int_norm);

    % remove overlap
    X_xcr(:,idx) = X_xcr_p(2*pg_int/dc+1:3*pg_int/dc);
end

%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE ENERGY PEAK TO AVERAGE RATIO (PAR)
%%%%%%%%%%%%%%%%%%%%%%%
a = max(abs(X_xcr),[],1);
b = sum(abs(X_xcr), 1);
X_xcr_par = a ./ (b / (pg_int/dc));

%%%%%%%%%%%%%%%%%%%%%%%
% DISCARD DOPPLER PERIODS UNTIL SUFFICIENT ENERGY PAR IS DETECTED
% note: This removes "start of recording" glitches
%%%%%%%%%%%%%%%%%%%%%%%

k = find(X_xcr_par > 5);

if (length(k) <= 0)
    error('No Doppler periods with sufficient Peak-to-Average energy');
end

if ((length(k) + 1) < n_dpp)
    error('Need at least 2 Doppler periods with sufficient energy');
end

% discard doppler periods before sufficient energy PAR is detected
X_xcr = X_xcr(:,k(1):end);

%%%%%%%%%%%%%%%%%%%%%%%
% EQUALIZE
%%%%%%%%%%%%%%%%%%%%%%%

% TODO

[m,idx] = max(abs(X_xcr(:,1)));

% capture 10us before and after
spread = floor (2e-6 / ts_int);

a = X_xcr(idx-spread+1:idx+spread,1);
b = fftshift(fft(a));
%figure;
%plot(abs(b));

%%%%%%%%%%%%%%%%%%%%%%%
% MEASURE POWER
%%%%%%%%%%%%%%%%%%%%%%%

% TODO


% % normalize detection by finding the highest correlation
% mc = max(abs(xcr_int));
% 
% % find first peak
% idx = find(xcr_int > (0.5 * mc));
% 
% % extract peaks
% xcr_max = abs(xcr_int(idx(1):2048*10:end));
% 
% % normalize
% xcr_norm = xcr_max * (1/pg);
% 
% % convert to power
% xcr_pwr = xcr_norm .^ 2;
% xcr_pwr_db = 10 * log10(xcr_pwr);
% 
% % compute path loss
% tx_dbW = tx_dbm - 30;
% pl = tx_dbW - xcr_pwr_db + txant_db + rxant_db;

figure;
% subplot(2,2,1);
% plot(xcr_pwr_db);
% ylabel('db-Watts');
% subplot(2,2,2);
% plot(pl);
% ylabel('dB attenuation')
% ylim([50 150]);
% title('path loss');

%subplot(2,2,3);
%surf(abs(X_xcr(1.445e4:1.480e4,2:end)),'EdgeColor','none');
%image(abs(X_xcr(1.445e4:1.480e4,2:end)));

% find loudest multi-path
[m,idx] = max(abs(X_xcr(:,1)));

% capture 10us before and after
spread = floor (2e-6 / ts_int);
a = X_xcr(idx-spread+1:idx+spread,:);
t_mp = length(a(:,1)) / (fs * usf);
t = ts_int:ts_int:t_mp;

y_mp = length(a(1,:));
y = (1/dplr):(1/dplr):(y_mp/dplr);

t2d = t' * ones(1,length(a(1,:)));

y2d = ones(length(a(:,1)),1) * y;

plot(t, abs(a));
xlabel('time (s)');
ylabel('power (W)');
title('Delay Spreads');

figure;
surf(t2d,y2d,abs(a),'EdgeColor','none');
xlabel('spread (s)');
ylabel('time (s)');
zlabel('power (W)');
title('Delay Spreads');









