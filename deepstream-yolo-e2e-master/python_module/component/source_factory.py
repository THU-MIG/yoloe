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
import gi
import os
import configparser
import subprocess
from python_module.component.yt_factory import get_yt_uri
from python_module.common.platform_info import PlatformInfo

gi.require_version('Gst', '1.0')
from gi.repository import Gst


def cb_newpad(decodebin, decoder_src_pad,data):

    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    if not caps:        
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    platform_info= PlatformInfo()
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if (name.find("nvv4l2decoder") != -1):
        if (platform_info.is_integrated_gpu()):
            Object.set_property("enable-max-performance", True)
            Object.set_property("drop-frame-interval", 0)
            Object.set_property("num-extra-surfaces", 0)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)

 
def create_source_bin(index,uri):
    print("Creating source bin")

    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    
    uri_decode_bin.set_property("uri",uri)
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def parse_media_source(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    media_entries = []

    for section in config.sections():
        enable = config.getint(section, 'enable')
        if enable == 1:
            type = config.get(section, 'type')
            url = config.get(section, 'url')
            uri = config.get(section, 'url')

            if not os.path.isfile(url) and type == 'file':
                print(f"File not found: {url}")
            if type == "file":
                uri = f"file://{url}"
            if type == 'youtube':
                uri = get_yt_uri(url)
            media_entries.append((type,url,uri))
    return media_entries
