#!/bin/bash
dnf update -y
dnf install -y docker nginx
systemctl enable --now docker
usermod -aG docker ec2-user

docker pull usmanasif10p/ml-ops
docker run -d --name model-server -p 8000:8000 --restart unless-stopped usmanasif10p/ml-ops

tee /etc/nginx/conf.d/model.conf > /dev/null << 'NGINX'
server {
    listen 80;
    server_name _;
    location / {
        proxy_pass http://localhost:8000;
    }
}
NGINX

systemctl enable --now nginx