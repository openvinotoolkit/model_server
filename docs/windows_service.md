# OpenVINO Model Server as service on Windows {#ovms_docs_deploying_server_service}

## This document describes installation and usage of OpenVINO Model Server as a service on Windows

### 1. Download the windows zip package and install it using baremetal instructions

**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.

## Deploying the model server service

### Install service
```bat
sc create ovms binPath= "%cd%\ovms\ovms.exe --rest_port 8000 --config_path %cd%\models\config.json --log_level INFO --log_path %cd%\ovms_server.log" DisplayName= "OpenVino Model Server"
```

## For windows package with python this step is required to set service description and add python dependency to PATH for the service
```bat
ovms install
```

## Optionally set your own service description
```bat
sc description ovms "Hosts models and makes them accessible to software components over standard network protocols."
```

### Start the service with arguments passed during the `sc create` service installation
```bat
sc start ovms
```

### Start the service with new ovms arguments
```bat
sc start ovms --rest_port 8000 --config_path %cd%\models\config.json --log_level INFO --log_path %cd%\ovms_server.log
```

### Stop the service
```bat
sc stop ovms
```

### Stop the service
```bat
sc delete ovms
```