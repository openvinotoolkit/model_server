.PHONY: centos
all: centos

centos:
	docker build -f Dockerfile.centos -t ovms --build-arg http_proxy=$(http_proxy) --build-arg https_proxy="$(https_proxy)" .

