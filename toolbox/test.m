profile on;
TS_example('fft');
profile viewer;
p = profile('info');
profsave(p, 'profile_results');
profile off;
profile report;
