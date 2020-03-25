# Performance environment preparation 

## Requiremnets
* vim 
* docker version >= 19.03

**Docker Community Edition Information**

To install Docker Community Edition (CE), follow: official Docker instructions.
The build process requires access to the docker command. Remember to add your user to the docker group by running: 
```
sudo usermod -aG docker [user]
```
 if the user has not been previously added.
For more information, refer to the Post-install Docker guide.

## Nginx configuration
**1. Prepare *nginx* configuration directory:**  
```
mkdir -p nginx/conf.d
```
**2. Create file *default.conf*:**  
```
vim nginx/conf.d/default.conf
```
and fill it in with the following content:
```

server {
    listen       80 http2;
    server_name  localhost;
    access_log  /var/log/nginx/<directory>/access.log  upstream_time;

    location / {
        #root   /usr/share/nginx/html;
        #index  index.html index.htm;
        grpc_pass grpc://<grpc_address>:<grpc_port>;
    }
    
    location = /50x.html {
        root   /usr/share/nginx/html;
    }
}

server {
    listen       8080;
    server_name  localhost;

    location / {
        stub_status;
        access_log off;
    }
}
```
where:

 - `<directory>` - fill with any directory name,
 - `<grpc_address>` - url to grpc service,
 - `<grpc_port>`- port to grpc service.

**3. Create file *nginx.conf*:**  
```
vim nginx/nginx.conf
```
and fill it with the following content:
```
user  nginx;
worker_processes  1;
error_log  /var/log/nginx/error.log warn;
pid        /var/run/nginx.pid;

events {
    worker_connections  1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    log_format upstream_time '$remote_addr - $remote_user [$time_local] '
                         '"$request" $status $body_bytes_sent '
                         '"$http_referer" "$http_user_agent" '
                         'rt=$request_time uct="$upstream_connect_time" uht="$upstream_header_time" urt="$upstream_response_time"';


    sendfile        on;
    keepalive_timeout  65;
    include /etc/nginx/conf.d/*.conf;
}
```
## Nginx-exporter configuration
**1. Prepare *nginx-exporter* configuration directory:**  
```
mkdir nginx-exporter/
```
**2. Create file *config.hcl*:**  
```
vim nginx-exporter/config.hcl
```
and fill it with the following content:
```
listen {
  port = <port>
}

namespace "nginx" {
  format = "$remote_addr - $remote_user [$time_local] \"$request\" $status $body_bytes_sent \"$http_referer\" \"$http_user_agent\" rt=$request_time uct=\"$upstream_connect_time\" uht=\"$upstream_header_time\" urt=\"$upstream_response_time\""
  source {
    files = [
      "/var/log/nginx/<directory>/access.log"
    ]
  }
}
```
where:

 - `<nginx-exporter-port>` - nginx-exporter metrics port - fill with any 
available port number, i.e. `4040`,
- `<directory>` - fill with the same name as in *Nginx configuration* section.

## Volume preparation - [details]([https://docs.docker.com/storage/volumes/](https://docs.docker.com/storage/volumes/))
Create docker volume to store data permanently:
```
docker volume create nginx_logs
```
## Running nginx
Run *nginx* using *docker*:
```
docker run --name nginx -d -p 80:80 -p 8080:8080 -v /path/to/nginx/nginx.conf:/etc/nginx/nginx.conf -v /path/to/nginx/conf.d:/etc/nginx/conf.d -v nginx_logs:/var/log/nginx/<directory>/ nginx
```
where:

 - `<directory>` -  fill with the same name as in *Nginx configuration* section.

## Running nginx-exporter
Run *nginx-exporter* using *docker*:
```
docker run --name nginx-exporter -v nginx_logs:/var/log/nginx/<directory>/ -v /path/to/nginx-exporter/config.hcl:/config.hcl -p <nginx-exporter-port>:<nginx-exporter-port> -d --entrypoint=./prometheus-nginxlog-exporter quay.io/martinhelmich/prometheus-nginxlog-exporter -config-file /config.hcl
```
where:
 - `<directory>` -  fill with the same name as in *Nginx configuration* section,
 - `<nginx-exporter-port>` - fill with the same port number as in *Nginx-exporter configuration* section.

## Check if *nginx* and *nginx-exporter* properly configured and running

*1. Download metrics from *nginx-exporter* :*
```
curl <nginx-address>:<nginx-exporter-port>/metrics 2>/dev/null | grep nginx
```
   where:
* `<nginx-address>`- fill with address of machine where nginx is running,
* `<nginx-exporter-port>`- fill with the same port number as in *Nginx-exporter configuration* section.

Output should be similar with the following:
```
# HELP nginx_parse_errors_total Total number of log file lines that could not be parsed
# TYPE nginx_parse_errors_total counter
nginx_parse_errors_total 0
```

 *2. Run inference as in <getting_started_url> .*
 
 *3. Download metrics again from *nginx-exporter*.*
 
Output should be similar with the following:
```
# HELP nginx_http_response_count_total Amount of processed HTTP requests
# TYPE nginx_http_response_count_total counter
nginx_http_response_count_total{method="POST",status="200"} 91
# HELP nginx_http_response_size_bytes Total amount of transferred bytes
# TYPE nginx_http_response_size_bytes counter
nginx_http_response_size_bytes{method="POST",status="200"} 367025
# HELP nginx_http_response_time_seconds Time needed by NGINX to handle requests
# TYPE nginx_http_response_time_seconds summary
nginx_http_response_time_seconds_sum{method="POST",status="200"} 2.4699999999999993
nginx_http_response_time_seconds_count{method="POST",status="200"} 91
# HELP nginx_http_response_time_seconds_hist Time needed by NGINX to handle requests
# TYPE nginx_http_response_time_seconds_hist histogram
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="0.005"} 0
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="0.01"} 1
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="0.025"} 51
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="0.05"} 91
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="0.1"} 91
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="0.25"} 91
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="0.5"} 91
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="1"} 91
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="2.5"} 91
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="5"} 91
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="10"} 91
nginx_http_response_time_seconds_hist_bucket{method="POST",status="200",le="+Inf"} 91
nginx_http_response_time_seconds_hist_sum{method="POST",status="200"} 2.4699999999999993
nginx_http_response_time_seconds_hist_count{method="POST",status="200"} 91
# HELP nginx_http_upstream_time_seconds Time needed by upstream servers to handle requests
# TYPE nginx_http_upstream_time_seconds summary
nginx_http_upstream_time_seconds{method="POST",status="200",quantile="0.5"} NaN
nginx_http_upstream_time_seconds{method="POST",status="200",quantile="0.9"} NaN
nginx_http_upstream_time_seconds{method="POST",status="200",quantile="0.99"} NaN
nginx_http_upstream_time_seconds_sum{method="POST",status="200"} 2.470999999999999
nginx_http_upstream_time_seconds_count{method="POST",status="200"} 91
# HELP nginx_http_upstream_time_seconds_hist Time needed by upstream servers to handle requests
# TYPE nginx_http_upstream_time_seconds_hist histogram
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="0.005"} 0
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="0.01"} 1
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="0.025"} 51
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="0.05"} 91
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="0.1"} 91
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="0.25"} 91
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="0.5"} 91
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="1"} 91
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="2.5"} 91
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="5"} 91
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="10"} 91
nginx_http_upstream_time_seconds_hist_bucket{method="POST",status="200",le="+Inf"} 91
nginx_http_upstream_time_seconds_hist_sum{method="POST",status="200"} 2.470999999999999
nginx_http_upstream_time_seconds_hist_count{method="POST",status="200"} 91
# HELP nginx_parse_errors_total Total number of log file lines that could not be parsed
# TYPE nginx_parse_errors_total counter
nginx_parse_errors_total 0
```