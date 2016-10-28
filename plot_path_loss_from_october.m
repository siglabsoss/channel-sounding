
figure

x = 100:50:1500;
IEEE80216 = PL_IEEE80216d(2450e6,x);

plot(x, IEEE80216, 'k')
plot(parsed2450(1:63,10),parsed2450(1:63,11),'or')
hold on
plot(parsed2450(64:93,10),parsed2450(64:93,11),'ob')
hold on
plot(parsed2450(94:144,10),parsed2450(94:144,11),'og')
hold on
plot(x, IEEE80216, 'k')

ylim([0 200])

legend('10/19/2016 (Wed)', '10/20/2016 (Thu)', '10/21/2016 (Fri)', '802.16 path loss')
legend('Location','southeast')
xlabel('range (m)')
ylabel('path loss (dB)')
title('Indoor-to-Outdoor-to-Indoor RF Propagation Path Loss 2450MHz');
hline(180,'m:','detection threshold');

grid on

figure

IEEE80216 = PL_IEEE80216d(915e6,x);

plot(parsed915(1:63,10),parsed915(1:63,11),'or')
hold on
plot(parsed915(64:93,10),parsed915(64:93,11),'ob')
hold on
plot(parsed915(94:144,10),parsed915(94:144,11),'og')
hold on
plot(x, IEEE80216, 'k')
ylabel('path loss (dB)')
xlabel('range (m)')
ylim([0 200])
title('Indoor-to-Outdoor-to-Indoor RF Propagation Path Loss 915MHz');
grid on
legend('Location','southeast')
legend('10/19/2016 (Wed)', '10/20/2016 (Thu)', '10/21/2016 (Fri)', '802.16 path loss')
legend('Location','southeast')
hline(180,'m:','detection threshold');