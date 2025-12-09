@REM Docker API (for metrics) + FaaS Gateway + Redis Intermediate Storage + Redis Metrics Storage
ssh -L 2376:127.0.0.1:2375 -L 5001:127.0.0.1:5000 djesus@146.193.41.126 -N