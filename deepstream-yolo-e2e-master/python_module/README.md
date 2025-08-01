 
## General Configuration Settings
The application can be configured using the config/config.init file. Below are the key settings you can modify:


### Configuration Parameters

- **OUTPUT_DIRECTORY**: This is the directory where files will be saved when using the `--output file` option. Set the path as needed.
  
- **OUTPUT_PREFIX**: This parameter specifies the prefix for the output file names. You can customize it according to your requirements.

- **RTSP_PORT**: This is the port used for the RTSP stream. The default value is `8554`.

- **RTSP_FACTORY**: This represents the path for the RTSP stream. For example, setting it to `/live` allows for streaming under this path.

- **RTSP_UDPSYNC**: This is the internal port used by DeepStream to connect to the RTSP server. The default value is 8255.

**RTSP URL Format** <br>When constructing the RTSP URL, it will always follow this format:
```
rtsp://<server_ip>:<RTSP_PORT><RTSP_FACTORY>
```
For example, if you use the default settings, the URL would be:

```bash
rtsp://<server_ip>:8554/live
```

The application can be configured using the [`config/config.ini`](config/config.ini) file. <br> Below are the key settings you can modify:




```ini
[Settings]
MUXER_BATCH_TIMEOUT_USEC = 33000
MUXER_OUTPUT_WIDTH = 1280
MUXER_OUTPUT_HEIGHT = 720

TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720

OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 1

OUTPUT_DIRECTORY = ./videos_output
OUTPUT_PREFIX = deepstream_out

RTSP_PORT = 8554
RTSP_FACTORY = /live
RTSP_UDPSYNC = 8255
```


