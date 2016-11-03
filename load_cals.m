
load Z:\161016-1_backpack_calibration\cal_rx_step2450.mat
load Z:\161016-1_backpack_calibration\cal_rx_step915.mat
load z:\161017-1_backpack_channel_sounding\tx.mat

[gain_error_db_2450, ~] = cal_procedure(cal_rx_step2450, 1, 1, 1.625e6);
gain_error_db_2450_at_setting = gain_error_db_2450 + 10;
gain_error_2450_at_setting = 10^(gain_error_db_2450_at_setting/20);

close all

[gain_error_db_915, ~] = cal_procedure(cal_rx_step915, 1, 1, 1.625e6);
gain_error_db_915_at_setting = gain_error_db_915 + 3;
gain_error_915_at_setting = 10^(gain_error_db_915_at_setting/20);

close all
