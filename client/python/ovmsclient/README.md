# ovmsclient - a lightweight client for OpenVINO Model Server and TensorFlow Serving

The [ovmsclient](https://pypi.org/project/ovmsclient/) package is a lightweight alternative for [tensorflow-serving-api](https://pypi.org/project/tensorflow-serving-api/). Contrary to `tensorflow-serving-api`, `ovmsclient` does not come with `tensorflow` as a dependency. This way the package is way smaller, but still capable of interacting with the serving. 

Download ovmsclient with:

```bash
pip3 install ovmsclient
```

[Learn more about the client package...](lib)


## Try with the samples

If you're looking for a quick way to try `ovmsclient` out or want to look up how things look in the code, see [ovmsclient samples](samples).

## Use in Docker container

There are also Dockerfiles available that prepare Docker image with `ovmsclient` installed and ready to use.
Simply run `docker build` with the Dockerfile of your choice to get the minimal image:
- [Ubuntu 20.04 based image](Dockerfile.ubuntu)
- [UBI 8.7 based image](Dockerfile.redhat)

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server/client/python/ovmsclient

docker build --no-cache -f Dockerfile.ubuntu .

docker build --no-cache -f Dockerfile.redhat .
```