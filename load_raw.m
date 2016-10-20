function out = load_raw(file_name)

f = dir(file_name);
len = f.bytes;

fid = fopen(file_name, 'r');

b = fread(fid, len/4, 'single');

out = complex(b(1:2:end), b(2:2:end));

