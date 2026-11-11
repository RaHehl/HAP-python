r"""A camera with the HomeKit Secure Video 17.99 features, for integration tests.

Developer Preview, experimental: Apple has not finalised HKSV 17.99 and
the services may change with any iOS or tvOS release.

The camera offers:

- H.265 multi-tier live view for local controllers
  (``MultiTierMediaDelegate``, one tier per stream),
- WebRTC live view for remote controllers through a home hub
  (``WebRTCMediaDelegate``),
- HKSV recordings with a prebuffer (``RecordingMediaDelegate``), triggered
  by a motion sensor that fires every ``--motion-interval`` seconds and on
  ``kill -USR1 <pid>``.

All three pull the same ffmpeg sources through one ``MediaWorker``: a
channel watched by several sessions is pulled once. The video is copied, so
the input must be H.265. Recordings are transcoded to H.264.

Prerequisites:

- ffmpeg with libx265 (test source), libx264 (recordings, classic live
  view) and libopus (audio),
- this branch of HAP-python, e.g. ``pip install -e .`` in its checkout.

``--highest``, ``--high``, ``--medium`` and ``--low`` each take one H.265
stream as ``URL@WIDTHxHEIGHT`` and announce it to HomeKit as the tier of that
video quality. The guide asks for at least three per sensor: High (the full
sensor resolution, 4K, 2K or 1080p, also recorded), Medium (1080p, 720p for
a 1080p camera) and Low (360p or 240p). Highest is an optional 4K stream
next to a 2K High. ``--high`` is required, with fewer than three the library
warns, which lets you see what a hub does with them. One URL can serve
several qualities. The controller picks a tier per session and switches
between them. The lowest stream gives the snapshots and the classic live
view. ``test`` as URL encodes a test picture (testsrc2 and a sine tone) at
start, which is the default for High, Medium and Low::

    python accessories/camera_hksv_main.py

    python accessories/camera_hksv_main.py \
        --high rtsp://user:pass@camera/high@1920x1080 \
        --medium rtsp://user:pass@camera/medium@1280x720 \
        --low rtsp://user:pass@camera/low@640x360 --fps 30

Each can also come from ``HKSV_HIGHEST``, ``HKSV_HIGH``, ``HKSV_MEDIUM`` and
``HKSV_LOW``. The sizes and ``--fps`` must be what the streams deliver, they
are advertised as is.

Pairing: the log prints the setup code and a QR code (with
``pip install HAP-python[QRCode]``). In the Home app tap +, Add Accessory,
More options, pick "HKSV Camera" and enter the code. The pairing is stored
in ``--persist`` (default ``camera_hksv.state``); delete it to pair again.

What to look at in the Home app:

- Live view at home (Wi-Fi): an iPhone or iPad with iOS 27 streams H.265
  over the Multi-Tier RTP service. The log shows "Multi-tier stream" lines;
  zooming in or a slow network switches tiers.
- Live view away from home (Wi-Fi off, cellular): needs a home hub (Apple TV
  or HomePod) with tvOS/HomePod OS 27. The remote viewer streams over WebRTC
  through the hub; the log shows "WebRTC" lines.
- Recordings: set the camera to "Stream & Allow Recording" in its settings.
  The hub arms the recording delegate, which buffers the main stream. Each
  motion event starts a recording that begins ``--prebuffer`` ms early.
  Observed with iOS and tvOS 27: hubs do not upload HDS recordings of a
  camera that also has the Multi-Tier service, so with the default setup the
  recording runs (see the log) but never appears in the Home app. Start with
  ``--classic`` to test recordings that do appear: the camera then has no
  Multi-Tier or WebRTC service, and live view is classic H.264 RTP
  transcoded by ffmpeg.

Changing ``--classic`` changes the services of a paired camera: remove it
from the Home app and delete the state file first.
"""

import argparse
import asyncio
import contextlib
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from typing import Any
from uuid import UUID, uuid5

from pyhap import camera, hksv3, util
from pyhap.accessory_driver import AccessoryDriver
from pyhap.camera_multi_tier import MultiTierConfig
from pyhap.camera_webrtc import WebRTCConfig
from pyhap.media.ffmpeg import FFmpegSourceProvider
from pyhap.media.multi_tier_delegate import MultiTierMediaDelegate
from pyhap.media.recording_delegate import RecordingMediaDelegate
from pyhap.media.webrtc_delegate import WebRTCMediaDelegate
from pyhap.media.worker import MediaWorker

logger = logging.getLogger("camera_hksv")

SENSOR_UUID = UUID("6f0c3a52-8f1e-4d39-9a57-0d1b2c3e4f50")
# Average and peak kbps of the guide's bit rate table, by pixel count.
GUIDE_BITRATES = (
    (640 * 480, 180, 190),
    (1440 * 1080, 768, 800),
    (1920 * 1080, 1700, 1800),
    (2560 * 1440, 2800, 3000),
    (3840 * 2160, 4500, 5000),
)
# Seconds a motion event stays on.
MOTION_DURATION = 10
TEST_CLIP_SECONDS = 20
# Classic RTP live view: the input transcoded to H.264, video only.
CLASSIC_STREAM_CMD = (
    "{ffmpeg} -hide_banner -nostdin -loglevel warning {input} -map 0:v:0 -an "
    "-c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p "
    "-profile:v baseline -r {{fps}} -vf scale={{width}}:{{height}} "
    "-b:v {{v_max_bitrate}}k -bufsize {{v_max_bitrate}}k "
    "-maxrate {{v_max_bitrate}}k -payload_type {{v_payload_type}} "
    "-ssrc {{v_ssrc}} -f rtp -srtp_out_suite AES_CM_128_HMAC_SHA1_80 "
    "-srtp_out_params {{v_srtp_key}} "
    "srtp://{{address}}:{{v_port}}?rtcpport={{v_port}}&"
    "localrtcpport={{v_port}}&pkt_size=1316"
)


# The test picture when no stream is given.
DEFAULT_STREAMS = {
    hksv3.VideoQuality.HIGH: "test@1920x1080",
    hksv3.VideoQuality.MEDIUM: "test@1280x720",
    hksv3.VideoQuality.LOW: "test@640x360",
}


def parse_stream(value: str) -> tuple[str, tuple[int, int]]:
    """Parse URL@WIDTHxHEIGHT, the size after the last @."""
    url, _, size = value.rpartition("@")
    width, _, height = size.lower().partition("x")
    if not url or not width.isdigit() or not height.isdigit():
        raise argparse.ArgumentTypeError(f"expected URL@WIDTHxHEIGHT, got {value!r}")
    return url, (int(width), int(height))


def bitrates(width: int, height: int) -> tuple[int, int]:
    """Average and peak kbps the guide allows for a frame size."""
    pixels = width * height
    for limit, average, peak in GUIDE_BITRATES:
        if pixels <= limit:
            return average, peak
    return GUIDE_BITRATES[-1][1:]


def input_args(url: str) -> list[str]:
    """ffmpeg input arguments of a camera URL."""
    if url.startswith(("rtsp://", "rtsps://")):
        return ["-rtsp_transport", "tcp", "-i", url]
    return ["-i", url]


def make_test_clip(
    ffmpeg: str, encoder: str, size: tuple[int, int], fps: int, directory: str
) -> list[str]:
    """Encode a looping H.265 test clip, return its ffmpeg input arguments.

    MPEG-TS keeps the Annex B parameter sets the copy needs.
    """
    width, height = size
    path = os.path.join(directory, f"test-{width}x{height}.ts")
    gop = 2 * fps
    if encoder == "libx265":
        params = f"keyint={gop}:min-keyint={gop}:scenecut=0:log-level=error"
        video = ["-preset", "ultrafast", "-x265-params", params]
    else:
        video = ["-g", str(gop)]
    logger.info("Encoding a %dx%d H.265 test clip with %s", width, height, encoder)
    subprocess.run(  # noqa: S603 - our own arguments, no shell
        [
            *(ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"),
            *("-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}"),
            *("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"),
            *("-t", str(TEST_CLIP_SECONDS), "-map", "0:v", "-map", "1:a"),
            *("-c:v", encoder, *video, "-pix_fmt", "yuv420p"),
            *("-c:a", "aac", "-b:a", "64k", "-f", "mpegts", path),
        ],
        check=True,
    )
    return ["-re", "-stream_loop", "-1", "-i", path]


class HKSVCamera(camera.Camera):
    """A camera whose live view and recordings come from one source pool."""

    def __init__(self, args: argparse.Namespace, driver: AccessoryDriver) -> None:
        """Build the sources, the delegates and the camera options."""
        self.motion_interval = args.motion_interval
        self._motion_task: asyncio.Task[None] | None = None
        self._tmp = tempfile.mkdtemp(prefix="camera-hksv-")
        # Each quality announced with its stream, best first.
        tiers = {
            quality: (
                make_test_clip(args.ffmpeg, args.encoder, size, args.fps, self._tmp)
                if url == "test"
                else input_args(url),
                size,
            )
            for quality, (url, size) in sorted(args.streams.items())
        }
        sub = tiers[max(tiers)][0]
        self.snapshot_input = sub
        sizes = {quality: size for quality, (_, size) in tiers.items()}
        main_size = tiers[hksv3.VideoQuality.HIGH][1]

        # One worker loop and one pool: each channel is pulled once for all
        # local and remote sessions and the recording.
        self.worker = MediaWorker()
        self.sources = FFmpegSourceProvider(
            {quality: ffmpeg_input for quality, (ffmpeg_input, _) in tiers.items()},
            worker=self.worker,
            ffmpeg=args.ffmpeg,
            audio=not args.no_audio,
        )
        width, height = tiers[min(tiers)][1]
        average, _peak = bitrates(*main_size)
        self.recording = RecordingMediaDelegate(
            self.sources,
            hksv3.VideoStreamTier(
                0, hksv3.VideoQuality.HIGH, average, *main_size, args.fps
            ),
            prebuffer_ms=args.prebuffer,
        )
        self.multi_tier = MultiTierMediaDelegate(self.sources)
        self.webrtc = WebRTCMediaDelegate(self.sources)

        options: dict[str, Any] = {
            # The classic RTP service: iOS 27 needs it next to the 17.99 ones.
            "video": {
                "codec": {
                    "profiles": [camera.VIDEO_CODEC_PARAM_PROFILE_ID_TYPES["BASELINE"]],
                    "levels": [
                        camera.VIDEO_CODEC_PARAM_LEVEL_TYPES["TYPE3_1"],
                        camera.VIDEO_CODEC_PARAM_LEVEL_TYPES["TYPE4_0"],
                    ],
                },
                "resolutions": [
                    [1920, 1080, 30],
                    [1280, 720, 30],
                    [640, 360, 30],
                    [480, 270, 30],
                    [320, 240, 15],  # Apple Watch
                    [320, 180, 30],
                ],
            },
            "audio": {"codecs": [{"type": "OPUS", "samplerate": 24}]},
            "srtp": True,
            "address": args.address or util.get_local_address(),
            "start_stream_cmd": CLASSIC_STREAM_CMD.format(
                ffmpeg=args.ffmpeg,
                input=" ".join(sub).replace("{", "{{").replace("}", "}}"),
            ),
            "recording": {"delegate": self.recording},
            "sensors": [
                hksv3.SensorConfiguration(
                    SENSOR_UUID,
                    width,
                    height,
                    video_streams=[
                        hksv3.VideoStreamCapability(
                            uuid5(SENSOR_UUID, quality.name),
                            quality,
                            *size,
                            args.fps,
                            *bitrates(*size),
                        )
                        for quality, size in sizes.items()
                    ],
                )
            ],
            "global_operating_mode": ["ManuallyDisabled"],
        }
        if not args.classic:
            options["multi_tier"] = MultiTierConfig(delegate=self.multi_tier)
            options["webrtc"] = WebRTCConfig(delegate=self.webrtc)
        super().__init__(options, driver, "HKSV Camera")

    async def run(self) -> None:
        """Start the media worker, the HDS listener and the motion events."""
        # Starting the worker takes longer than a controller waits for an offer.
        await self.worker.start()
        logger.info("Media worker running")
        await super().run()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGUSR1, self.trigger_motion)
        self._motion_task = loop.create_task(self._periodic_motion())

    async def _periodic_motion(self) -> None:
        if self.motion_interval <= 0:
            return
        while True:
            await asyncio.sleep(self.motion_interval)
            self.trigger_motion()

    def trigger_motion(self) -> None:
        """Fire a motion event, which starts a recording while recording is on."""
        logger.info(
            "Motion detected (recording %s)", "on" if self.recording_active else "off"
        )
        self.set_motion_detected(True)
        self.driver.loop.call_later(MOTION_DURATION, self.set_motion_detected, False)

    async def async_get_snapshot(self, image_size: dict[str, Any]) -> bytes:
        """A JPEG of the sub stream, the stock picture if ffmpeg fails."""
        process = await asyncio.create_subprocess_exec(
            *(self.sources.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error"),
            *self.snapshot_input,
            *("-frames:v", "1", "-vf", f"scale={image_size['image-width']}:-2"),
            *("-f", "image2", "-c:v", "mjpeg", "pipe:1"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            jpeg, _ = await asyncio.wait_for(process.communicate(), 8)
        except TimeoutError:
            process.kill()
            await process.wait()
            jpeg = b""
        return jpeg or self.get_snapshot(image_size)

    async def stop(self) -> None:
        """End the sessions, then release the sources and the worker."""
        with contextlib.suppress(NotImplementedError, RuntimeError):
            asyncio.get_running_loop().remove_signal_handler(signal.SIGUSR1)
        if self._motion_task is not None:
            self._motion_task.cancel()
        # Ends the sessions and disarms the recording delegate.
        await super().stop()
        await self.multi_tier.close()
        await self.webrtc.close()
        await self.sources.close()
        await self.worker.close()
        shutil.rmtree(self._tmp, ignore_errors=True)


def main() -> None:
    """Parse the arguments and run the camera until SIGINT or SIGTERM."""
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    for quality in hksv3.VideoQuality:
        name = quality.name.lower()
        parser.add_argument(
            f"--{name}",
            type=parse_stream,
            default=os.environ.get(f"HKSV_{quality.name}"),
            metavar="URL@WIDTHxHEIGHT",
            help=f"H.265 stream of the {name} tier, 'test' as URL for a test picture",
        )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--no-audio", action="store_true", help="the streams have no audio"
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help="no Multi-Tier and WebRTC, so recordings are uploaded",
    )
    parser.add_argument(
        "--prebuffer", type=int, default=4000, help="recording prebuffer in ms"
    )
    parser.add_argument(
        "--motion-interval",
        type=float,
        default=60,
        help="seconds between motion events, 0 for none",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--encoder", default="libx265", help="H.265 encoder of the test source"
    )
    parser.add_argument("--address", help="address to stream from (default: auto)")
    parser.add_argument("--port", type=int, default=51826)
    parser.add_argument("--persist", default="camera_hksv.state")
    args = parser.parse_args()
    args.streams = {
        quality: stream
        for quality in hksv3.VideoQuality
        if (stream := getattr(args, quality.name.lower())) is not None
    }
    if not args.streams:
        args.streams = {
            quality: parse_stream(value) for quality, value in DEFAULT_STREAMS.items()
        }
    if hksv3.VideoQuality.HIGH not in args.streams:
        parser.error("--high is required")

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    driver = AccessoryDriver(port=args.port, persist_file=args.persist)
    driver.add_accessory(accessory=HKSVCamera(args, driver))
    # SIGINT (KeyboardInterrupt) and SIGTERM stop the driver, which stops the
    # camera, the HAP server and the advertisement.
    signal.signal(signal.SIGTERM, driver.signal_handler)
    driver.start()


if __name__ == "__main__":
    main()
