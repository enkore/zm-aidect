# zm-aidect

-- A turnkey object (human, animal, vehicle) detection system for ZoneMinder --

How it works: zm-aidect runs alongside ZoneMinder and feeds images from ZoneMinder into a highly sophisticated
artifical intelligence (actually it's tinyYOLO-V4 and ML and not AI but marketing said otherwise). If the AI says something is there,
zm-aidect tells ZoneMinder to record it. Configuration happens through ZoneMinder itself, just like motion detection. Events are recorded
by ZoneMinder as per the usual.

zm-aidect uses images decoded by ZoneMinder, so there is no double decoding of streams, and also no increase in camera traffic.
zm-aidect works with any camera/video source configurable in ZoneMinder, not just IP cameras.

## Installation

### When using a Debian Bullseye or newer system

You're in luck - I make binaries for these.

### From source

* install opencv in whichever flavor you need (e.g. opencv-cuda package if you want to use CUDA)
* install Rust
* "cargo build --release"
* ???
* copy some files around I guess

## Configuration

Assuming the systemd service has been installed, you only need to enable (and start) zm-aidect for the monitors it
should run on (using `3` as the monitor ID; repeat these commands for every monitor you wish to use):

    # systemctl enable zm-aidect@3
    # systemctl start zm-aidect@3

    # systemctl status zm-aidect@3
    ● zm-aidect@3.service
         Loaded: loaded (/etc/systemd/system/zm-aidect@.service; enabled; vendor preset: enabled)
        Drop-In: /run/systemd/system/service.d
                 └─zzz-lxc-service.conf
         Active: active (running) since Fri 2022-06-10 22:51:56 CEST; 1h 6min ago
       Main PID: 293563 (zm-aidect)
          Tasks: 4 (limit: 19145)
         Memory: 60.7M
            CPU: 3.141s
         CGroup: /system.slice/system-zm\x2daidect.slice/zm-aidect@3.service
                 └─293563 /zm-aidect/zm-aidect 3

zm-aidect is pretty turnkey beyond this. You configure it ZoneMinder's web interface by adding a zone
named "aidect". Objects will be detected if within the zone. You can additionally tweak various settings by
adding them to the zone's name:

* Threshold=XX adjusts the confidence threshold for detection (0-100 %). The default is 50%.
* Size=XX adjusts the input size handed to the model. The default is 416 (pixels), 256 pixels works generally fine.
  128 and 192 requires the objects to be fairly large in the frame. Inference is sped up by the *square* of this number, e.g.
  256 is around 2.5x faster than 416, 128 can be up to 10x faster (depending on CPU thread count and/or if an accelerator is used).
* Classes=1,2,3,... sets which classes trigger detection. By default only humans and cars will be detected.
  See class names. Because the length of the zone name is limited, we can't use human-readable names here.
  The default is: 1,3,15,16,17 (persons, cars, birds, cats and dogs).
* MinArea=51600 filters detections by their area. In triggered events the area is indicated "aidect: Human (51.1%) 90x177 (=**15930**) at 440x385".
  Some things can persistently trigger medium-to-high confidence detections and filtering by area is a simple way to get rid of these.
  Alternatively, consider having the aidect zone not cover those patterns if they are static.
* FPS=XX sets the maximum analysis fps for zm-aidect and zm-aidect alone. The default is the analysis FPS set in the monitor,
  and if that isn't set zm-aidect will, just like ZoneMinder's own analysis, run as fast as possible to try and catch them all.
* Trigger=XX sets an alternative monitor ID for triggering. This is useful when evaluating zm-aidect, because
  you can attach zm-aidect to your normal substream monitor, but trigger events on a secondary nodect monitor so that
  you can compare whatever method you normally use and zm-aidect, without having to have two monitors decode
  the same stream. That's the only thing this option is good for.

For example:

    aidect Size=128 Threshold=40 FPS=5

Multiple "aidect" zones should not be added to a single monitor and aren't supported.

Changing settings in ZoneMinder will be reflected within a few seconds in zm-aidect; you don't need to systemctl-restart
it manually.

Making the zone smaller does not speed things up (except if it's really small), but can improve detection accuracy.
Note that the zone is embedded in a rectangular region, meaning that the ML model will see a rectangle fitting around
the zone; having long protrusions thus reduces effectiveness. Overall, the ML model used here is good enough even
for full-screen detection on very wide angle cameras. Don't sweat it. Zone placement and size matters A LOT less with
zm-aidect than it does with traditional motion detection.

### Testing changes

You can also run `zm-aidect --test <MONITOR-ID>`, which will go through the startup, perform a single inference
to ensure the process works, trigger an event, and exit. Some diagnostics will be printed as well, like if and which
hardware accelerator is used by zm-aidect. This can be used to confirm that the settings are applied as wanted.

Run `zm-aidect --event=12345` to have zm-aidect analyze the given event as-if it were watching live, using the current settings
of the monitor the event belongs to. Detections will be printed,  no triggering takes place.
This can be used to verify that aidect does (not) detect something you (don't) want to detect without getting up.

## Performance

Machine learning is very resource intensive. It *can* be done on CPUs, but it is vastly more CPU-intensive than
ZoneMinder's motion detection (~100x more CPU used). This means that 1.) if you use zm-aidect without a GPU or other
accelerator, you almost certainly will have to reduce the analysis FPS **a lot**. "Going from 10 fps to 1-2 fps"-lot.
2.) You will see **way** higher CPU load (and temperatures, power draw as well as possibly noise).

That being said, the fact that ML is *extremely* insensitive towards lighting changes, as well as rapid non-object motion
and very slow object motion (e.g. objects approaching head-on, slowly)¸ makes it worth it in my opinion. As far as I can tell
analysis running at just one or two fps is not a significant issue for detection. ML is also very good at rejecting noise,
probably in not-so-small part because the image is downscaled a ton.

Finally, this uses a general-purpose object detection model, which has been pre-trained on a generic dataset. Hardly optimal.
The model is designed to detect *eighty different classes* of objects, but even if you want to detect and capture cats and
stray dogs - you most likely use less than five classes. It seems a certainty to me that performance could be greatly improved
with a model tailored to and trained for this application.

### Performance at full-size (416x416)

Input image size: 1280x720

Network input size: 416x416 (default for tinyYOLO-v4)

Note that this testing has been performed like the application works in practice, i.e. the batch size is one,
with one inferrence being serially pre-processed, inferred and post-processed on the same thread. OpenCV can
parallelize only the inferrence itself (theoretically the RGB->NCHW conversion and the NMS suppression as well,
but that seems unlikely, the former is basically memcpy).

* System idle
  * CPU: 26-30 W
  * GPU: 13 W (P8)
* Ryzen 5600X using all cores, running at ~4.5 GHz with this load:
  * 17-18 ms per inference
  * 900 % CPU (~75 % utilization)
  * ~300 MB CPU memory
  * 100% PPT (83 W), core power ~60 W
* Ryzen 5600X using one thread (at ~4.7 GHz):
  * 70-72 ms per inference
  * ~300 MB CPU memory
  * 60% PPT (~50 W), core power 20-23 W
* 1080 Ti using CUDA/cuDNN and CV-managed threads:
  * 2.2-2.4 ms per inference
  * 75% core load
  * roughly 180/250 W
  * 140 % CPU (apparently CPU-limited)
  * CPU consumes around 50 W
  * Peak GPU memory use ~500 MB
  * Peak main memory use ~1.3 GB
* 1080 Ti with just one CPU thread:
  * 2.3-2.4 ms per inference
  * 71% core load
  * ~170 W, 500 MB GPU mem, ~1.3 GB CPU mem

GPU usage, power and memory numbers were taken from `nvidia-smi`, while the Ryzen power figures are taken from the SMU.

Conclusions: Unsurprisingly, the GPU is quite a lot quicker with far superior perf/W. Single-threaded inference
is very wasteful on the Ryzen due to the large I/O die and fabric power overhead (generally around ~20 W, which
for lightly threaded workloads on Ryzen means that less than 50 % of the power goes into the cores).

```
Ryzen R5 5600X 1T		14 Hz	50 W	  = 0.3 Hz/W
Ryzen R5 5600X All-core		58 Hz	83 W	  = 0.7 Hz/W
nVidia 1080 Ti			455 Hz	180+50 W  = 2 Hz/W
 - 3x better power efficiency, even including CPU power on the inefficient Zen 3 Ryzen
 - 8x higher performance
```

Another way to look at power efficiency is to consider the rise above idle. Because the 1080 Ti is much better at
having low-power modes for idle and low-intensity workloads (like P8, with a claimed, but also independently
verified ~13 W total board power), its power spread is larger than the Ryzen, which is generally quite poor about this
(the multi-die SKUs that is, the monolithically made Ryzen APUs are quite good, because they're a totally different
SoC architecture).

Regardless, the four year older GPU maintains a more than comfortable ~2.4x power efficiency lead, as you would expect.

```
Ryzen R5 5600X 1T		24 W	 = 0.6 Hz/W
Ryzen R5 5600X All-core		57 W	 = 1 Hz/W
nVidia 1080 Ti			167+24 W = 2.4 Hz/W
```

Moral of the story: Don't do ML inference on CPUs. At least not desktop parts. Maybe AVX-512 on an Intel 7 Xeon changes
the picture (a bit).

### Performance at reduced size (256x256)

tinyYOLO-v4 retains pretty good detection performance for large-ish objects at this resolution. Depending on the
exact application, 192x192 may still work satisfactorily as well.

```
5600X 1T			37 Hz	(2.6x) 50 W	  = 0.7 Hz/W
5600X All-core			139 Hz	(2.4x) 83 W	  = 1.7 Hz/W
1080 Ti				740 Hz	(1.6x) 140+45 W	  = 4 Hz/W
```

Efficiency gap shrinks (unsurprisingly: GPUs don't like small, one-off jobs). Performance scales almost perfectly in
accordance with the size reduction (416/256**2 = 2.65), at least when using one thread. Multi-threading diminishes
the impact a little bit; this gap would likely increase if you were to use more cores (per Amdahl). The GPU benefits
least from reducing the size, which I'd interpret as the overhead of submitting the small inference jobs. Throughput
should increase in a much more faithful manner if batching or multiple parallel processes are used.
