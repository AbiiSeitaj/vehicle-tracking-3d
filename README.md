Monocular 3D Bounding Box Tracking from Video

Python pipeline for vehicle detection, tracking and projected 3D cuboids from an uncalibrated, fixed-camera video. It uses YOLOv8n, ByteTrack and a flat-ground camera model.

No camera matrix or lens parameters are supplied to the pipeline. It estimates scene geometry from video evidence where possible and uses explicit priors where that evidence is insufficient. It does not recover unique metric 3D ground truth from one camera.

Example output
Example output. Fits are approximate and degrade on overlapping vehicles.

Setup

Tested on Windows with Python 3.14 and CPU inference. From the repository root:

py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-win-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe scripts/download_model.py

Use Python 3.14 for the tested dependency set. Other Python versions and operating systems have not been verified. requirements.txt lists the direct pinned dependencies; requirements-lock-win-cpu.txt pins the tested Windows CPU environment. No separate FFmpeg installation is required: imageio-ffmpeg supplies the executable used for video verification. Dependency and model downloads require network access.

This repository contains code only. It does not include any video, footage or model weights, and it does not link to any. You must provide your own video. Make sure you own the footage or have permission to use it.

Run on your own video

Use a fixed-camera clip (for example, a phone on a tripod or ledge). A moving or panning camera violates the geometry model. Put your clip in the data folder, which is created empty by the repository and ignored by Git, or pass any path:

.\.venv\Scripts\python.exe run_demo.py --input data/input.mp4 --output outputs/output.mp4 --snapshot outputs/output.jpg --max-frames 90
.\.venv\Scripts\python.exe scripts/verify_video.py outputs/output.mp4 --expected-frames 90

Start with a short run like the one above, then drop --max-frames to process the whole clip. Add --show for a live preview, and omit it when no graphical display is available. The first 45 frames are sampled for scene estimation, then the complete video is processed. Press Q or Escape to stop early. Output includes an annotated MP4, per-frame predictions and a metrics JSON. Live preview is not proof of real-time throughput.

Pipeline
detectors.py: pretrained YOLO vehicle detections, confidence and size filtering.
tracker.py: official Ultralytics ByteTrack association; unmatched tracks are not drawn.
geometry.py: scene estimate, ground-plane intersection, cuboid fitting and pose smoothing.
renderer.py: projected edges, IDs and uncertain-fit indicators.
pipeline.py: decoding, initialization, processing and output files.

Default inference size is 640, detection confidence is 0.10, and the minimum box height is 35 pixels at processed width 1280. This height cutoff is a size filter, not a physical distance threshold.

Geometry
1. Infer scene geometry and record the assumptions

Canny edges and Hough line segments from the initialization frames are grouped using RANSAC to find candidate road-line vanishing points. Two directions assumed to be orthogonal on the ground plane constrain focal length:

c = (image_width / 2, image_height / 2)
f^2 = -(v1 - c) dot (v2 - c)

Here v1 and v2 are image-space vanishing points and c is the assumed principal point. This relation requires square pixels, zero skew and orthogonal scene directions; line grouping alone cannot prove those directions are ground-plane directions. Weak or inconsistent hypotheses are rejected.

If line evidence is insufficient, detected vehicles provide approximate horizon votes:

horizon_y ~= box_bottom - camera_height * box_height / class_height
f = image_width / (2 * tan(horizontal_FOV / 2))

The horizon relation comes from comparing apparent vehicle height with the ground-contact displacement from the horizon. It approximates equal depth for those measurements, neglects footprint extent, and assumes class height and camera height. Border-truncated boxes are excluded from voting; median/MAD filtering reduces outlier influence. Repeated observations can be correlated. If usable vehicle evidence is also absent, a configured horizon prior is used.

Vehicle dimensions are fixed class priors. The horizon can use video evidence, but focal length, metric scale and vehicle dimensions are not independently recovered. Camera diagnostics record the selected evidence and assumptions.

2. Construct a camera and intersect the ground plane

World coordinates use X across the road, Y upward and Z along the road. The ground is Y = 0, and the camera center is C = (0, h, 0). Camera coordinates use X right, Y down and Z forward. R transforms world directions into camera coordinates.

K = [[f, 0, cx],
     [0, f, cy],
     [0, 0,  1]]

road_direction_camera = normalize(K^-1 * [vx, vy, 1])
ground_horizon_line   = K^-T * ground_normal_camera

With two accepted ground directions, their cross product gives the ground normal. Otherwise, a horizon prior constrains the normal to be perpendicular to the road direction. These directions define an orthonormal R; one vanishing point alone cannot independently determine focal length and camera roll. Lens distortion is not modeled.

For the bottom-center pixel p = (u, v, 1) of a detected box, the initial ground position is a ray-plane intersection:

r = R^T * K^-1 * p
lambda = -h / r_y
P_ground = C + lambda * r

This requires a forward intersection below the ground horizon. Bottom-center is only an approximate ground-contact observation, particularly with occlusion. It initializes the cuboid fit rather than fixing the final position. Ground points also induce the homography K * [R[:,0], R[:,2], -R*C] from (X, Z, 1) to image coordinates; the implementation uses the equivalent camera/ray formulation.

3. Fit and project one rigid cuboid

For class-prior length L, width W and height H, local corners are all combinations of x = +/-W/2, y = 0 or H, and z = +/-L/2. Rotate these corners about world Y by yaw and translate them to footprint center (X, 0, Z).

q = K * R * (P_corner - C)
pixel = (q_x / q_z, q_y / q_z)

SciPy bounded least squares with a robust loss fits (X, log(Z), yaw) by comparing the projected corner envelope with the detected 2D box. Box-edge errors are normalized by detection width and height; extra residuals penalize nonpositive corner depth and departure from the yaw prior. Image-border truncation uses an edge-censored envelope comparison. Dimensions remain fixed, and a single 2D box does not uniquely identify yaw, depth or physical dimensions.

All eight corners use the same K and R. This produces a perspective-consistent projection under the inferred camera model, rather than an arbitrary screen-space extrusion. It does not prove that the inferred camera or cuboid matches the physical scene.

4. Maintain temporal stability and reject invalid fits

ByteTrack supplies vehicle IDs. Each ID retains a pose for warm-start fitting; periodic multi-start fitting helps recover from a poor local solution. Accepted pose parameters are exponentially smoothed before reprojection, preserving a rigid cuboid instead of smoothing eight image points independently.

Corners behind the near plane and fits exceeding the normalized envelope-error gate are rejected. Rejected tracked detections remain visible as 2D boxes marked uncertain; unmatched tracks are not drawn. Smoothing reduces jitter but can introduce lag, and ID switches disrupt continuity. Stable appearance and a good projected-box fit are not measurements of physical 3D accuracy.

Results

Baseline measurements on an Intel Core i7-6820HQ CPU. None of the clips used below (the private test clip, bus and highway) are included in or linked from this repository:

Measurement	Result
Frames processed	743, the full readable source
Processing throughput	6.60 FPS
Median / p95 processing latency	137.54 / 242.70 ms
Accepted cuboid track-frame instances	1456 of 2122 (68.6%)

Both OpenCV and independent FFmpeg decoding produce 743 frames; container metadata reports 744. Timing includes the frame-processing loop and output writing, but excludes model initialization and scene estimation. Source playback is about 14.61 FPS, so this CPU run is not real time. GPU throughput has not been tested.

Sparse 2D evaluation uses AI-assisted visual annotations, class-agnostic IoU >= 0.5 and box height >= 35 pixels:

Subset	Annotated frames / objects	Detection precision / recall	Track recall	Sampled IDF1
Private test clip	8 / 27	91.7% / 81.5%	66.7%	60.9%
Bus	10 / 26	72.7% / 61.5%	26.9%	41.2%
Highway, initial	10 / 1	Undefined / 0%	0%	0%
Highway, extended	10 / 10	100% / 80%	40%	57.1%

The highway runs cover the first 300 and 1000 frames respectively, not all 2000. The extended subset is positive-enriched, not a random sample; keep its scores separate. Neither highway run produced accepted cuboids. Low resolution and moving cameras make these failure cases, not evidence of robust transfer.

Labels cover only sampled frames and visible 2D extents. No independently reviewed dense tracking or metric 3D ground truth is available. Compact results are in examples/results.json.

Tests and evaluation
.\.venv\Scripts\python.exe -m pytest -q

All 85 tests pass, covering geometry invariants, tracking, evaluation, configuration, video encoding and report paths. These tests verify implementation behavior, not real-world 3D accuracy.

The results above were measured on footage that is not distributed with this repository. The annotation files in scripts/validation_samples/ hold only my sampled box labels. To evaluate your own run, create annotations in the same JSON format and pass them to scripts/evaluate.py with your predictions file. Evaluation reports name the annotation subset, and a positive-enriched subset must not replace a sparse one. Input clips and generated reports stay local and are not committed.

Limitations
Not metric 3D. Focal length, camera height and vehicle dimensions are priors. A single uncalibrated camera cannot recover unique metric geometry, and the projected cuboids are not measurements of physical 3D accuracy.
Fixed-camera assumption. Moving-camera footage (bus, highway) violates the geometry model. Neither highway run produced accepted cuboids.
Partial rejection. 31.4% of tracked vehicle-frames on the private test clip failed the fit gate and are shown as uncertain 2D boxes.
Sparse evaluation. Results use AI-assisted annotations on a few sampled frames, in 2D only. There is no dense tracking or 3D ground truth.
Model simplifications. Lens distortion is not modeled, vehicle dimensions are fixed class priors, and smoothing can add lag. ID switches break continuity.
Performance and platform. 6.60 FPS on CPU is not real time, and GPU was not tested. Only Windows with Python 3.14 was verified.
License

See LICENSE.