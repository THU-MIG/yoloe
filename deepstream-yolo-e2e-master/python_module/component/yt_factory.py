"""
Creative Commons Attribution-NonCommercial 4.0 International License

You are free to share and adapt the material under the following terms:
- Attribution: Give appropriate credit.
- NonCommercial: Not for commercial use without permission.

For inquiries: levi.pereira@gmail.com
Repository: DeepStream / YOLO (https://github.com/levipereira/deepstream-yolo-e2e)
License: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""
import yt_dlp

# ℹ️ See help(yt_dlp.YoutubeDL) for a list of available options and public functions
ydl_opts = {}

def format_selector(ctx):
     # formats are already sorted worst to best
    formats = ctx.get('formats')[::-1]

    # acodec='none' means there is no audio
    best_video = next(f for f in formats
                      if f['vcodec'] != 'none' and f['acodec'] == 'none' and f['height'] <= 1080 and f['fps'] <= 30 )

    # These are the minimum required fields for a merged format
    yield {
        'format_id': f'{best_video["format_id"]}',
        'ext': best_video['ext'],
        'requested_formats': [best_video],
        # Must be + separated list of protocols
        'protocol': f'{best_video["protocol"]}'
    }


def get_yt_uri(url):

    ydl_opts = {
        'format': format_selector,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        #info_json = json.dumps(ydl.sanitize_info(info))
        uri  = info.get('requested_formats', [])[0].get('url')
    return uri
    