listen {
  port = <nginx_exporter_port>
}


namespace "<server_name>" {
  format = "$remote_addr - $remote_user [$time_local] \"$request\" $status $body_bytes_sent \"$http_referer\" \"$http_user_agent\" rt=$request_time uct=\"$upstream_connect_time\" uht=\"$upstream_header_time\" urt=\"$upstream_response_time\""
  source {
    files = [
      "/var/log/nginx/nginx_logs/<server_name>.log"
    ]
  }
}
