"""
Creative Commons Attribution-NonCommercial 4.0 International License

You are free to share and adapt the material under the following terms:
- Attribution: Give appropriate credit.
- NonCommercial: Not for commercial use without permission.

For inquiries: levi.pereira@gmail.com
Repository: DeepStream / YOLO (https://github.com/levipereira/deepstream-yolo-e2e)
License: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""

import configparser
import gi
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GstRtspServer, GstRtsp
from python_module.component.system_config import get_config
from python_module.common.platform_info import PlatformInfo

def create_rtsp_server():
    platform_info = PlatformInfo()
    config_values = get_config()
    rtsp_port_num = config_values['RTSP_PORT']
    rtsp_stream_end = config_values['RTSP_FACTORY']
    updsink_port_num = config_values['RTSP_UDPSYNC']
    if platform_info.is_jetson_nano_device():
        codec = 'H264'
    else:
        codec = 'H265'

    server = GstRtspServer.RTSPServer.new()
    auth = GstRtspServer.RTSPAuth()

    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_protocols(GstRtsp.RTSPLowerTrans.UDP |
                      GstRtsp.RTSPLowerTrans.TCP )
    factory.set_transport_mode(GstRtspServer.RTSPTransportMode.PLAY)
    factory.set_latency(1)
    factory.set_launch(
        '( udpsrc name=pay0  port=%d buffer-size=10485760  caps="application/x-rtp, media=video, clock-rate=90000, mtu=1300, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    
    server.get_mount_points().add_factory(rtsp_stream_end, factory)
    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://%s:%d%s ***\n\n" %
        ('localhost', rtsp_port_num, rtsp_stream_end))