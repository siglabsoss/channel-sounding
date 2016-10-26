ts_int = 1/6.25e6/5;


% index of the multipath averaging window we are computing
idx_mp = 1;

% RMS delay spread threshold (dB)
rms_thresh = 8;

% noise threshold in power
m = max(abs(multipath_vec(:,idx_mp)));
%ds_thresh = 0.2e-9
ds_thresh = m / (10^(rms_thresh/10));


% threshhold vector
multipath_vec_zidx = find(abs(multipath_vec(:,idx_mp)) < ds_thresh);
multipath_vec_thresh = multipath_vec(:, idx_mp);
multipath_vec_thresh(multipath_vec_zidx) = 0;

% find first multi-path
multipath_vec_thresh_pidx = find(abs(multipath_vec_thresh) > 0);
t_first_mp = multipath_vec_thresh_pidx(1) * ts_int;


t = (ts_int:ts_int:length(multipath_vec_thresh)*ts_int)' - t_first_mp;

figure; semilogy(t,[abs(multipath_vec_thresh) abs(multipath_vec(:,idx_mp))],'-o');xlabel('time (s)');

hline(ds_thresh,'r:','threshold');

tau = sum(abs(multipath_vec_thresh) .* t,1) / sum(abs(multipath_vec_thresh),1);
tau2 = sum(abs(multipath_vec_thresh) .* (t .^ 2),1) / sum(abs(multipath_vec_thresh),1);
sigma_tau = sqrt(tau2 - tau^2);

% maximum excess delay
%med = tau + sigma_tau
med = t(multipath_vec_thresh_pidx(end));

bc = 1/(50 * sigma_tau);

disp(sprintf('Mean Excess Delay = %0.1f(ns)\nRMS Delay Spread = %0.1f(ns)\nMaximum Excess Delay < %0.0fdB = %0.1f(ns)\nMaximum Coherence Bandwidth (>0.9 frequency correlation) = %0.0f(Hz)', tau*1e9, sigma_tau*1e9, rms_thresh, med*1e9, bc ));

