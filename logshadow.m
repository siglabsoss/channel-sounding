function out = logshadow(x, pl0, lamda,d0)

out = pl0 + 10*lamda*log10(x/d0);