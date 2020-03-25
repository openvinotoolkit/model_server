#!/bin/bash
echo "Removing nginx containers"
docker stop haproxy
docker stop nginx
docker stop nginx-exporter
docker rm haproxy
docker rm nginx
docker rm nginx-exporter

echo "Recreating volume nginx_logs"
docker volume rm nginx_logs
docker volume create nginx_logs

echo "Starting nginx container..."
docker run --name nginx -d -p <nginx_port>:<nginx_port> -v /root/OVMS-performance/nginx/nginx.conf:/etc/nginx/nginx.conf -v /root/OVMS-performance/nginx/conf.d:/etc/nginx/conf.d -v nginx_logs:/var/log/nginx/nginx_logs/ nginx

echo "Starting nginx-exporter container..."
docker run --name nginx-exporter -v nginx_logs:/var/log/nginx/nginx_logs/ -v /root/OVMS-performance/nginx-exporter/config.hcl:/config.hcl -p <nginx_exporter_port>:<nginx_exporter_port> -d --entrypoint=./prometheus-nginxlog-exporter quay.io/martinhelmich/prometheus-nginxlog-exporter -config-file /config.hcl -config-file /config.hcl

echo "Starting haproxy container..."
docker run -d -p <haproxy_port>:<haproxy_port> -p <haproxy_stats_port>:<haproxy_stats_port> --name haproxy --network haproxy_network -v /root/OVMS-performance/haproxy:/usr/local/etc/haproxy:ro haproxy:2.1.3

echo "Checking the environment"
docker ps
echo "Metrics for ovms"
curl localhost:4040/metrics 2>/dev/null | grep ovms
echo "Metrics for tfs"
curl localhost:4040/metrics 2>/dev/null | grep tfs
echo "Metrics for min_server"
curl localhost:4040/metrics 2>/dev/null | grep min_server
echo "Environment is running"
