# OpenVINO Model Server as service on Windows {#ovms_docs_deploying_server_service}

## This document describes installation and usage of OpenVINO Model Server as a service on Windows

### Download the windows zip package and install it using baremetal instructions

**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.

## Deploying the model server service

### Install service
```bat
sc create ovms binPath= "%cd%\ovms\ovms.exe --rest_port 8000 --config_path %cd%\models\config.json --log_level INFO --log_path %cd%\ovms_server.log" DisplayName= "OpenVino Model Server"
```

### For windows package with python this step is required to set service description and add python dependency to PATH for the service
```bat
ovms install
```

#### Optionally set your own service description
```bat
sc description ovms "Hosts models and makes them accessible to software components over standard network protocols."
```

### Start the service with arguments passed during the `sc create ovms` service installation
```bat
sc start ovms
```

### Optionally start the service with new ovms arguments
```bat
sc start ovms --rest_port 8000 --config_path %cd%\models\config.json --log_level INFO --log_path %cd%\ovms_server.log
```

### Stop the service
```bat
sc stop ovms
```

### Delete the service
```bat
sc delete ovms
```

## Service status gui
You can monitor the service status gui in the native services windows application.
Start it by writing services in the windows start search bar.
![Service Status](windows_service1.jpg)

## Service registry settings
You can monitor the service settings in the native windows registry editor application.
Start it by writing regedit in the windows start search bar.
![Service Status](windows_service_registry1.jpg)

## Service log and events
You can monitor the detailed service log with the --log_level [INFO,DEBUG,TRACE] option during the service creation and tje --log_path path_to_file paramter.
Additionally you can review the ket service events and errors in the native windows event viewer application.
Start it by writing event viewer in the windows start search bar.
![Service Status](windows_service_events1.jpg)
