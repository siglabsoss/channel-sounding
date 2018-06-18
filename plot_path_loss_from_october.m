
figure

load results.mat
calc_log_distance

x = 100:50:1500;
IEEE80216 = PL_IEEE80216d(2450e6,x);

plot(x, IEEE80216, 'k')
plot(parsed2450(:,10),parsed2450(:,11),'ok')
hold on
%plot(x, logshadow(x,pl0_2450,lambda2450,d0_2450), 'r');
hold on
plot(x, IEEE80216, '--k')


ylim([0 200])

legend('SigFi path loss', ...
    sprintf('SigFi Log-Shadow Model (lamda=%0.2f, std=%0.2f)',1,1), ...%lambda2450, dev2450), ...
    '802.16 path loss(reference)')
legend('Location','southeast')
xlabel('range (m)')
ylabel('path loss (dB)')
title('Indoor-to-Outdoor-to-Indoor RF Propagation Path Loss 2450MHz (Menlo Park, California)');
hline(180,'m:','detection threshold');

grid on

figure

IEEE80216 = PL_IEEE80216d(915e6,x);

plot(parsed915(:,10),parsed915(:,11),'ok')
hold on
%plot(x, logshadow(x,pl0_915,lambda915,d0_915), 'r');
hold on
plot(x, IEEE80216, '--k')
ylabel('path loss (dB)')
xlabel('range (m)')
ylim([0 200])
%title('Indoor-to-Outdoor-to-Indoor RF Propagation Path Loss 915MHz (Menlo Park, California)');
grid on
legend('SigFi path loss', ...
    sprintf('SigFi Log-Shadow Model (lamda=%0.2f, std=%0.2f)',1,1), ...%lambda915, dev915), ...
    '802.16 path loss(reference)')
legend('Location','southeast')
hline(180,'m:','detection threshold');