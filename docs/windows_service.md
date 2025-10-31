# OpenVINO Model Server as service on Windows (preview functionality) {#ovms_docs_deploying_server_service}

## This document describes installation and usage of OpenVINO Model Server as a service on Windows

### Download the windows zip package and install it using baremetal instructions

**Required:** OpenVINO Model Server package - see [deployment instructions](../../../docs/deploying_server_baremetal.md) for details.

## Starting the model server service

### Start the service with arguments passed during the `.\ovms\install_ovms_service.bat` service installation
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

## Demo
You can find a RAG demo with the service in the [demo documentation](../demos/continuous_batching/rag/README.md) in the service section.

## Service interaction
You can change the default config.json location from c:\models to you own path. Make sure you are passing the full path to the directory.
```bat
.\ovms\install_ovms_service.bat c:\my_own_path\to_directory
```

### List models
```bat
ovms --model_repository_path c:\models --list_models
```

### Pull models
```bat
ovms --pull --model_repository_path c:\models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation
```

### Start a model by adding it to the config.json
```bat
ovms --add_to_config c:\models --model_name OpenVINO/Qwen3-8B-int4-ov --model_path OpenVINO/Qwen3-8B-int4-ov
```

### Stop a model by removing it from the config.json
```bat
ovms --remove_from_config c:\models --model_name OpenVINO/Qwen3-8B-int4-ov
```

## Service status gui
You can monitor the service status gui in the native services windows application.
Start it by writing services in the windows start search bar.
![Service Status](windows_service1.jpg)

## Service registry settings
You can monitor the service settings in the native windows registry editor application.
Start it by writing regedit in the windows start search bar.
The service settings after installation are available at the `Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Services\ovms` path:
![Service Status](windows_service_registry1.jpg)

## Service log and events
You can monitor the detailed service log with the --log_level [INFO,DEBUG,TRACE] option during the service creation and the --log_path path_to_file parameter.

Additionally you can review the service events and errors in the native windows event viewer application.
Start it by writing event viewer in the windows start search bar.
The service events are visible in the `Windows Logs -> System` tab.
![Service Status](windows_service_events1.jpg)
