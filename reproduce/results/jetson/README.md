# Jetson Nano benchmark records

`bench.txt` is the console record of the benchmark runs.
`bench_parsed.json` is the same content parsed into per-model,
per-session fields.

Each file holds two sessions. The paper reports the first. The second is
included so a reader can check whether the timing replicates: it does, with mean
latency agreeing to within about half a millisecond on every model that
compiled.

## A note on the GPU utilisation field

The on-die utilisation counter on this board does not report reliably. In the
second session it returns `0.0` for every run, and elsewhere it repeats the same
value across different architectures. The field is therefore indicative only and
nothing in the paper rests on it.

Every other quantity comes from the same capture and behaves as expected:
latency, RAM, temperature and power all vary across models and agree between the
two sessions.

## Layout

Per model, per session:

- `latency`: `mean`, `std`, `median`, `p95`, `p99`, `min`, `max` in milliseconds,
  and `fps`
- `system`: GPU and CPU utilisation, RAM used and total in MB, mean and peak
  temperature in degrees Celsius, GPU and total power in milliwatts

Timing protocol: batch size 1, images held in RAM, CUDA-synchronised, a warm-up
pass on real images before the timed runs, five runs of 112 images per session.
The timed region wraps the framework's prediction call, which encloses its
internal preprocessing, the forward pass and non-maximum suppression. Frames per
second is the image count divided by total wall-clock time for the run, so it
additionally includes the per-frame telemetry sampling that sits outside the
timed region; the two denominators differ deliberately.
