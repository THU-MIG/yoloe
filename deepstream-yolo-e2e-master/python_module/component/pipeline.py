"""
Creative Commons Attribution-NonCommercial 4.0 International License

You are free to share and adapt the material under the following terms:
- Attribution: Give appropriate credit.
- NonCommercial: Not for commercial use without permission.

For inquiries: levi.pereira@gmail.com
Repository: DeepStream / YOLO (https://github.com/levipereira/deepstream-yolo-e2e)
License: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""

import sys
import os
import math
from datetime import datetime
import configparser
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst,GLib

from python_module.common.FPS import PERF_DATA
from python_module.component.source_factory import create_source_bin , parse_media_source
from python_module.component.rtsp_server import create_rtsp_server
from python_module.component.probes import sink_pad_buffer_probe, osd_sink_pad_buffer_probe
from python_module.component.pre_process import pre_process
from python_module.common.bus_call import bus_call
from python_module.common.utils import create_dynamic_labels
from python_module.component.system_config import get_config
from python_module.common.platform_info import PlatformInfo


# Function to create the pipeline
def create_pipeline(args, model_type):
    platform_info = PlatformInfo()
    Gst.init(None)
    stream_output=args.output.upper()
    config_values = get_config()

    
    output_file_path=None
    media_sources = parse_media_source('config/python_app/media.ini')

    number_sources = len(media_sources)
    if number_sources == 0:
        print("No active media sources found. Exiting...")
        sys.exit(1)
    perf_data = PERF_DATA(number_sources)



    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        return None

    elements = {
        "streammux": ("nvstreammux", "stream-muxer"),
        "pgie": ("nvinfer", "primary-inference"),
    }
    elements["tracker"] = ("nvtracker", "tracker")
    if stream_output == "SILENT":
        elements["sink"] = ("fakesink", "fakesink")

    if stream_output in ("FILE", "RTSP", "DISPLAY"):
        elements["nvvidconv_tiler"] = ("nvvideoconvert", "nvvidconv_tiler")
        elements["filter_tiler"] = ("capsfilter", "filter_tiler")
        elements["nvtiler"] = ("nvmultistreamtiler", "nvtiler")
        elements["nvosd"] = ("nvdsosd", "onscreendisplay")
        if stream_output in ("FILE","RTSP"):
            elements["nvvidconv_encoder"] = ("nvvideoconvert", "nvvidconv_encoder")
            elements["filter_encoder"] = ("capsfilter", "filter_encoder")
            if platform_info.is_jetson_nano_device():   ## nano does not have hardware encoder
                elements["encoder"] = ("x264enc", "encoder")
                elements["codeparser"] = ("h264parse", "h264-parser")
            else:
                elements["encoder"] = ("nvv4l2h265enc", "encoder")
                elements["codeparser"] = ("h265parse", "h265-parser")
            if stream_output  == "FILE":
                elements["container"] = ("matroskamux", "muxer")
                elements["sink"] = ("filesink", "file-sink")
            if stream_output  == "RTSP":
                if platform_info.is_jetson_nano_device():
                    elements["rtppay"] = ("rtph264pay", "rtppay")
                else:
                    elements["rtppay"] = ("rtph265pay", "rtppay")
                elements["sink"] = ("udpsink", "udpsink")
        else:
            if platform_info.is_jetson_device():
                elements["sink"] = ("nv3dsink", "nvvideo-renderer") 
            else:
                elements["sink"] = ("nveglglessink", "nvvideo-renderer")    

    
    ## Create Elements
    for name, val in elements.items():
        element = Gst.ElementFactory.make(val[0], val[1])
        if not element:
            sys.stderr.write(f"Unable to create {name}\n")
            return None
        elements[name] = element

    ## Configure Elements
    if os.environ.get('USE_NEW_NVSTREAMMUX') != 'yes':
        elements["streammux"].set_property('width', config_values['MUXER_OUTPUT_WIDTH'])
        elements["streammux"].set_property('height', config_values['MUXER_OUTPUT_HEIGHT'])
        elements["streammux"].set_property('batched-push-timeout', config_values['MUXER_BATCH_TIMEOUT_USEC'])
    elements["streammux"].set_property('batch-size', number_sources)

    if model_type == 'det':
        pgie_conf_file="/apps/deepstream-yolo-e2e/config/pgie/config_pgie_yolo_det.txt"
    elif model_type == 'seg':
        pgie_conf_file="/apps/deepstream-yolo-e2e/config/pgie/config_pgie_yolo_seg.txt"
    else:
        sys.stderr.write(f"Model Type not supported {model_type}\n")
        exit
    
    elements["pgie"].set_property('config-file-path', pgie_conf_file )


    elements["tracker"].set_property('tracker-width', 640)
    elements["tracker"].set_property('tracker-height', 384)
    elements["tracker"].set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    elements["tracker"].set_property('ll-config-file', '/apps/deepstream-yolo-e2e/config/tracker/config_tracker_NvDCF_perf.yml')
    elements["tracker"].set_property('display-tracking-id', 0)
    if platform_info.is_jetson_device():
        elements["tracker"].set_property('compute-hw', 2)
        

    if stream_output == "SILENT":
        elements["sink"].set_property('enable-last-sample', 0)
        elements["sink"].set_property('sync', 0)
    
    if stream_output in ("FILE", "RTSP", "DISPLAY"):
        elements["filter_tiler"].set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        tiler_rows=int(math.sqrt(number_sources))
        tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
        elements["nvtiler"].set_property("rows",tiler_rows)
        elements["nvtiler"].set_property("columns",tiler_columns)
        elements["nvtiler"].set_property("width", config_values['TILED_OUTPUT_WIDTH'])
        elements["nvtiler"].set_property("height", config_values['TILED_OUTPUT_HEIGHT'])

        elements["nvosd"].set_property('process-mode',config_values['OSD_PROCESS_MODE'])
        elements["nvosd"].set_property('display-text',config_values['OSD_DISPLAY_TEXT'])
        if model_type == 'det':
            elements["nvosd"].set_property('display-bbox', 1)
        if model_type == 'seg':
            elements["nvosd"].set_property('display-bbox', 0)
            elements["nvosd"].set_property('display-mask', 1)


        if stream_output in ("FILE", "RTSP"):
            if platform_info.is_jetson_device():
                elements["nvvidconv_tiler"].set_property('copy-hw', 2) 
                elements["nvvidconv_tiler"].set_property('compute-hw', 2) 
                elements["nvvidconv_encoder"].set_property('copy-hw', 2)
                elements["nvvidconv_encoder"].set_property('compute-hw', 2)
            
            if platform_info.is_jetson_nano_device():
                elements["filter_encoder"].set_property("caps", Gst.Caps.from_string("video/x-raw, format=I420"))
            else:
                elements["filter_encoder"].set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
            

            if platform_info.is_jetson_device() and not platform_info.is_jetson_nano_device() : 
                elements["encoder"].set_property('maxperf-enable', True) 
                elements["encoder"].set_property('preset-level', 1)
            elif platform_info.is_jetson_nano_device():
                elements["encoder"].set_property('bitrate', 2000000) 
                elements["encoder"].set_property('speed-preset', 'ultrafast') 
                elements["encoder"].set_property('tune', 'zerolatency') 
            elif not platform_info.is_platform_aarch64():
                elements["encoder"].set_property('tuning-info-id', 2) 
                elements["encoder"].set_property('control-rate', 2) 
            
            if stream_output == "FILE":
                output_directory = config_values['OUTPUT_DIRECTORY']   
                output_prefix =  config_values['OUTPUT_PREFIX']  

                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                output_file_name = f"{output_prefix}_{timestamp}.mp4"
                output_file_path = os.path.join(output_directory, output_file_name)

                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)

                elements["sink"].set_property('location', output_file_path)
                elements["sink"].set_property('sync', 0)
            
            if stream_output == "RTSP":
                elements["sink"].set_property('host', "127.0.0.1")
                elements["sink"].set_property('port', config_values['RTSP_UDPSYNC'])
                elements["sink"].set_property('async', False)
                elements["sink"].set_property('sync', 1)
            

    ## Add Elements to Pipeline

    for element in elements.values():
        pipeline.add(element)


    ## Create Sources 
    for index, (media, url, uri) in enumerate(media_sources):
        print(f"Source: {url}. Creating source_bin {index} \n ")
        source_bin = create_source_bin(index, uri)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % index
        sinkpad = elements["streammux"].request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)


    elements["streammux"].link(elements["pgie"])
    elements["pgie"].link(elements["tracker"])

    if stream_output == "SILENT":
        element_probe = elements["tracker"]
        elements["tracker"].link(elements["sink"])

    if stream_output in ("FILE", "RTSP", "DISPLAY"):
        element_probe = elements["nvtiler"]
        elements["tracker"].link(elements["nvvidconv_tiler"])
        elements["nvvidconv_tiler"].link(elements["filter_tiler"])
        elements["filter_tiler"].link(elements["nvtiler"])
        elements["nvtiler"].link(elements["nvosd"])
        if stream_output in ("FILE", "RTSP"):
            elements["nvosd"].link(elements["nvvidconv_encoder"])
            elements["nvvidconv_encoder"].link(elements["filter_encoder"])
            elements["filter_encoder"].link(elements["encoder"])
            if stream_output == "FILE":
                elements["encoder"].link(elements["codeparser"])
                elements["codeparser"].link(elements["container"])
                elements["container"].link(elements["sink"])
            if stream_output == "RTSP":
                elements["encoder"].link(elements["rtppay"])
                elements["rtppay"].link(elements["sink"])
        
        else:
            elements["nvosd"].link(elements["sink"])
    
    dynamic_labels = create_dynamic_labels(pgie_conf_file)

    return pipeline , element_probe, perf_data, output_file_path, dynamic_labels , number_sources


# Function to run the pipeline
def run_pipeline(args):
    model_type = pre_process(args.output)
    if args.output == "rtsp":
        create_rtsp_server()

    pipeline, element_probe, perf_data, output_file_path, dynamic_labels, number_sources  = create_pipeline(args, model_type)
    if not pipeline:
        sys.stderr.write("Failed to create pipeline\n")
        return

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    elementsinkpad = element_probe.get_static_pad("sink")
    if not elementsinkpad:
        sys.stderr.write("Unable to get sink pad of nvosd\n")
        return
    elementsinkpad.add_probe(Gst.PadProbeType.BUFFER, sink_pad_buffer_probe, 0, perf_data)
    
    osd = element_probe.get_static_pad("sink")
    if not osd:
        sys.stderr.write("Unable to get sink pad of nvosd\n")
        return
    osd.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0, dynamic_labels, number_sources)

    GLib.timeout_add(5000, perf_data.perf_print_callback)

    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)
    if args.output == "file":
        print(f"File output :  {output_file_path}") 