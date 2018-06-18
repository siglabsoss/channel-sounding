n = 100;
target = 20e6; % bps
bw = 20e6; % Hz
N = bw * 2 * 300 * 1.38e-23; % (bw * F * T * kB) Watts
eirp = 0.25; % W

% Log-Distance Shadow Model
r = 1500; % m
lamda = 2.64;
sigma = 11.28; % dB
d0 = 108.7; % m
pl0 = 134.9; % dB
% r = 1500; % ms
% lamda = 4.66;
% sigma = 11.28; % dB
% d0 = 108.7; % m
% pl0 = 122.2; % dB


clear C_AVE;
for idx = 1:50
    %% RANDOMPLY POSITION NODES
    % node 1 is base station (0,0)
    rng('shuffle');
    pos = zeros(n,2);
    for i = 2:n
        while 1
            [pos(i,1), pos(i,2)] = pol2cart(rand*2*pi,randi(r));
            d = sqrt((pos(i,1)-pos(1:i-1,1)).^2 + (pos(i,2)-pos(1:i-1,2)).^2);
            if min(d) > 50
                break
            end
        end
    end

    d = sqrt((pos(:,1)-pos(:,1)').^2 + (pos(:,2)-pos(:,2)').^2);
    d(d==0) = inf;

    % log-distance path loss and capacity between nodes
    pl_db = pl0 + 10 * lamda * log10(d/d0) + randn(n,n) * sigma; % dB
    pl_linear = 10.^(-pl_db/10);


    %% RECEIVE
    S = eirp * pl_linear;
    c_DIRECT = bw * log2(1 + S/N);

    c_DIRECT2 = min(permute(c_DIRECT .* ones(n,n,n),[1 3 2]),...
              permute(c_DIRECT .* ones(n,n,n),[3 2 1]));
    c_DIRECT2 = max(c_DIRECT2,[],3);

    %% AMPLIFY (METHOD 1)
    % Beaulieu, Norman C., and Jeremiah Hu. "A noise reduction
    % amplify-and-forward relay protocol for distributed spatial
    % diversity." IEEE communications letters 10.11 (2006).
    %
    % noise is reduced by a factor of two
    % note: Much better noise reduction techniques may be employed. The
    %       point of this model is to be as conservative as possible.
    gain = eirp./(S+N/2);

    REGEN = gain .* S .* ones(n,n,n);
    REGEN(isnan(REGEN)) = 0;
    REGEN = permute(REGEN,[1 3 2]);

    %% FORWARD
    pl_l = permute(pl_linear .* ones(n,n,n),[1 2 3]);
    S_AF = pl_l .* REGEN;
    S_AF_SUBSET = sort(S_AF,3);
    S_AF_SUBSET = S_AF_SUBSET(:,:,end-5:end); % only use the 10 best relays
    S_AF = sum(S_AF_SUBSET.^0.5,3).^2;
    C_AF = bw * log2(1 + S_AF/N);

    %% NEXT HOP
    C_D = min(permute(C_AF .* ones(n,n,n),[1 3 2]),...
              permute(C_AF .* ones(n,n,n),[3 2 1]));
    C_D = max(C_D,[],3);

    C_AVE(:,:,idx) = C_D;

end

%% PLOTS
close all

cm_snr = [linspace(0.2,0.5,128) linspace(0,0,128);...
    linspace(0,0,256);...
    linspace(0,0,128) linspace(0.5,1,128)]';
cm_cap = [linspace(0,1,256);...
    linspace(0,1,256);...
    linspace(0,1,256)]';

figure;
scaler = 1.1;
imagesc(newdelhi,'XData',[-scaler*r scaler*r], 'YData', [-scaler*r scaler*r])
colormap gray
hold on
plot(pos(:,1),pos(:,2),'.g','markers',20);
title('device locations');
xlim([-r*scaler scaler*r]);
ylim([-scaler*r scaler*r]);
hold off

figure;
hndl = imagesc(c_DIRECT/1e6);
hndl.AlphaData = 1-triu(ones(n));
%title('one hop channel capacity');
xlabel('Alice node');
ylabel('Bob node');
colormap hot;
c = colorbar;
c.Label.String = 'Mbps';
caxis([0 target/1e6]);

figure;
hndl = imagesc(c_DIRECT2/1e6);
hndl.AlphaData = 1-triu(ones(n));
%title({'two hop channel capacity','best relay selection method'});
xlabel('Alice node');
ylabel('Bob node');
colormap hot;
c = colorbar;
c.Label.String = 'Mbps';
caxis([0 target/1e6]);

figure;
imagesc(C_AF/1e6);
%title({'two hop AF channel capacity','noise reduction method 1'});
xlabel('Alice node');
ylabel('Bob node');
colormap hot;
c = colorbar;
c.Label.String = 'Mbps';
caxis([0 target/1e6]);

figure;
imagesc(C_D/1e6);
%title({'four hop AF channel capacity','noise reduction method 1'});
xlabel('Alice node');
ylabel('Bob node');
colormap hot;
c = colorbar;
c.Label.String = 'Mbps';
caxis([0 target/1e6]);

C_AVE(C_AVE > target) = target;

hist_data = mean(C_AVE(:,1,:),3);
hist_data(:,2) = mean(C_AVE(1,:,:),3)';

figure;
hist(hist_data/1e6,linspace(0,target/1e6,21));
xlim([0 target/1e6])
%title({'four hop subscriber performance'});
legend('subscriber uplink','subscriber downlink');
xlabel('Mbps');
ylabel('number of subscribers');



%autoArrangeFigures;