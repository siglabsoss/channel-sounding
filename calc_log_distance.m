

% grab good samples
idx = find(parsed915(:,11)> 0);
parsed915 = parsed915(idx,:);

idx = find(parsed2450(:,11)> 0);
parsed2450 = parsed2450(idx,:);


% get reference path loss
idx = find((parsed915(:,10) > 90) & (parsed915(:,10) < 110));
d0_915 = mean(parsed915(idx,10));
pl0_915 = mean(parsed915(idx,11));

idx = find((parsed2450(:,10) > 90) & (parsed2450(:,10) < 110));
d0_2450 = mean(parsed2450(idx,10));
pl0_2450 = mean(parsed2450(idx,11));


% fit log-distance shadow model
func = @(lamda,d) (pl0_915 + 10*lamda*log10(d/d0_915));
lambda915 = nlinfit(parsed915(:,10),parsed915(:,11),func,2)

func = @(lamda,d) (pl0_2450 + 10*lamda*log10(d/d0_2450));
lambda2450 = nlinfit(parsed2450(:,10),parsed2450(:,11),func,2)


% calculate variance
dev915 = mean(((pl0_915 + 10 * lambda915 * log10(parsed915(:,10)./d0_915)) - parsed915(:,11)).^2)^0.5
dev2450 = mean(((pl0_2450 + 10 * lambda2450 * log10(parsed2450(:,10)./d0_2450)) - parsed2450(:,11)).^2)^0.5
