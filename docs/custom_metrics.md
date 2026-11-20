# Custom metrics in C++ calculators {#ovms_docs_custom_metrics}

OVMS supports custom counters, gauges, and histograms in C++ MediaPipe
calculators. These metrics are exposed on the existing `/metrics` endpoint.
For general metrics endpoint setup and built-in metrics, see the
[metrics documentation](./metrics.md).

Calculators can register **custom counters, gauges, and histograms** and update
them during processing:

```cpp
auto events = metrics->getOrRegisterCounter("custom_events_total", "Events processed.");
events->increment();       // Add one.
events->increment(3.0);    // Add a batch.

auto queueDepth = metrics->getOrRegisterGauge("custom_queue_depth", "Queued requests.");
queueDepth->set(4);
queueDepth->add(-1);

auto latency = metrics->getOrRegisterHistogram(
    "custom_request_duration_seconds", "Request duration.", {0.01, 0.1, 1.0});
latency->observe(0.08);
```

`metrics` is an OVMS-provided `std::shared_ptr<ovms::CalculatorMetrics>`, scoped
to one graph and one named calculator node. Each returned shared pointer is a
thread-safe metric handle, not a local copy of the value. OVMS stores values in
the shared Prometheus registry and exposes them on the existing `/metrics`
endpoint. No separate cache, exporter, background thread or scrape callback is
necessary.

This is a **C++ extension API**: implementing new metric logic requires
compiling the calculator and linking it into OVMS; a graph file alone cannot
define arbitrary metric logic. It does not expose a Python metrics API.

## 1. Enable metrics and add the side packet

Enable the normal metrics endpoint using `--rest_port 8000 --metrics_enable`, or
the equivalent monitoring configuration. Then add this input side packet to the
calculator node in its graph:

```protobuf
node {
    name: "my_node"
    calculator: "MyCalculator"
    input_side_packet: "CALCULATOR_METRICS:calculator_metrics"
    # Other inputs, outputs and calculator options...
}
```

OVMS supplies the packet; do not configure an external side-packet value.
Use the tag and packet name exactly as shown (`CALCULATOR_METRICS:calculator_metrics`).
The node must have a stable, non-empty `name`. Each custom series automatically
gets `name="<graph name>"` and `node="<calculator node name>"` labels.

## 2. Register custom metrics in the calculator

Include `src/metrics/calculator_metrics.hpp` and depend on
`//src/metrics:libovmsmetrics` in the calculator's Bazel target.
Declare the side-packet type in `GetContract`, acquire the node's scope in `Open`,
and retain the returned handle as a calculator member:

```cpp
std::shared_ptr<ovms::CalculatorCounter> events;
std::shared_ptr<ovms::CalculatorGauge> queueDepth;
std::shared_ptr<ovms::CalculatorHistogram> latency;

static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
    // Keep the calculator's existing input/output contract here.
    cc->InputSidePackets()
        .Tag(ovms::CALCULATOR_METRICS_TAG)
        .Set<ovms::CalculatorMetricsMap>();
    return absl::OkStatus();
}

absl::Status Open(mediapipe::CalculatorContext* cc) override {
    const auto& scopes = cc->InputSidePackets()
        .Tag(ovms::CALCULATOR_METRICS_TAG)
        .Get<ovms::CalculatorMetricsMap>();
    const auto scope = scopes.find(cc->NodeName());
    RET_CHECK(scope != scopes.end()) << "Missing calculator metrics scope";
    try {
        events = scope->second->getOrRegisterCounter(
            "custom_events_total", "Events processed.");
        queueDepth = scope->second->getOrRegisterGauge(
            "custom_queue_depth", "Queued requests.");
        latency = scope->second->getOrRegisterHistogram(
            "custom_request_duration_seconds", "Request duration.",
            {0.01, 0.1, 1.0});
    } catch (const std::exception& e) {
        return absl::InvalidArgumentError(e.what());
    }
    return absl::OkStatus();
}
```

Use the normal MediaPipe headers for these methods, plus `<memory>` and
`<exception>`. For an optional integration, guard both contract declaration and
`Open` lookup with `cc->InputSidePackets().HasTag(ovms::CALCULATOR_METRICS_TAG)`.

Registration is idempotent: identical registrations from concurrent calculator
instances in the same graph/node return the same handle. Register in `Open`,
not for every packet. Different nodes or graphs get independent series.

## 3. Update during processing

Update the retained handles during processing:

```cpp
events->increment();
```

To count multiple events in a packet, call
`events->increment(static_cast<double>(eventCount))`. Use gauges with `set` or
`add` (accepting positive, zero, or negative finite values), and record histogram
samples with `observe`. Histogram bucket boundaries must be finite and strictly
increasing, and the same metric name must use identical boundaries everywhere.
No registry lookup, metric-name construction or registration is needed in
`Process`. Scrapes do not reset counters or histograms; use Prometheus `rate()`
or `increase()` to calculate counter changes over time.

## Naming, configuration and lifetime

- All names must start with `custom_` and contain only ASCII letters, digits
  and underscores. Counters must also end with `_total`; gauges and histograms
  use names that describe the measured quantity. Every metric needs non-empty
  help text. The `custom_` prefix keeps application metrics separate from
  built-in `ovms_` metrics.
- The same metric name must have identical type, help text, and (for histograms)
  bucket boundaries across nodes and graphs. Invalid names and conflicting
  definitions are logged and throw exceptions; do not suppress those errors.
  Non-finite gauge values or histogram observations, and negative or non-finite
  counter increments, are logged as warnings and ignored.
- Metric names must be fixed in calculator code. Do not construct names from
  words, users, requests or timestamps. This initial API intentionally exposes
  only the fixed graph/node labels to keep cardinality bounded.
- Custom metrics are controlled by the global metrics-enable setting and the
  calculator's own registration logic, **not** by the built-in `metrics_list`.
  Do not put custom names in that list: its parser accepts built-in names only.
  Custom metrics still work when a list selects a subset of built-in metrics.
- With metrics disabled, registration returns a valid no-op handle, so processing
  need not contain null checks. Names and values are still validated.
- Registration logs an INFO message once per metric and graph/node scope, not
  for each request. When metrics are disabled or the registry is unavailable,
  the message indicates that the handle is a no-op. A graph reload creates a
  new scope and can log the registration again.
- Metrics persist across requests and calculator `Close` calls. Do not remove
  them in `Close`: other calculator instances can share the series.
- The metric registry owns the series and keeps them until the server stops.
  Graph unload, idle-unload and reload do not remove or reset them: a calculator
  that registers the same metric after a reload continues the existing series,
  and counters and histograms keep their values. Handles retained by a request
  that started before the reload still update that same series.
- Because series outlive the graph, a series of a retired graph or a renamed
  node stays exported with its last value. Set gauges back to their idle value
  in `Close` if a stale value would be misleading.
