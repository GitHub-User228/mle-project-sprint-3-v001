# Monitoring

This file describes the monitoring system used to monitor the ML microservice.
A dashboard consists of 9 different panels:
- `CPU Usage`: Shows the CPU usage of the microservice over time window.
- `Memory Usage`: Shows the memory usage of the microservice over time window.
- `Error Rate`: Shows the error rate of the microservice over time window for different types of errors.
- `Error Increase`: Shows the error increase of the microservice over time window for different types of errors.
- `Price Prediction Quantiles (total)`: Shows the quantiles of the price prediction over all time.
- `Price Prediction Quantiles (over time window)`: Shows the quantiles of the price prediction over the time window
- `Prediction Request Duration Quantiles (total)`: Shows the quantiles of the prediction request processing time over all time.
- `Prediction Request Duration Quantiles (over time window)`: Shows the quantiles of the prediction request processing time over the time window
- `Prediction Request Rate`: Shows the rate of the prediction requests over time window.

## Several layer metrics have been selected for monitoring:

### Infrastrutural Metrics:
- `process_cpu_seconds_total`
    - **type**: Counter
    - **description**: Used to measure the CPU usage of the microservice.
    - **what for**: Used to monitor the CPU usage of the microservice in order to check if the CPU usage is not too high.
    - **dashboard**: Used in the `CPU Usage` time series panel, where the rate over a time window is taken in order to calculate the usage change in %.
- `process_virtual_memory_bytes`
    - **type**: Counter
    - **description**: Measures the virtual memory size used by the microservice.
    - **what for**: Can help in understanding the overall memory footprint of the process and can be used for capacity planning and resource allocation decisions
    - **dashboard**: Used in the `Memory Usage` time series panel, where the rate over a time window is taken in order to calculate the change.
- `process_resident_memory_bytes`
    - **type**: Counter
    - **description**: Measures the resident memory size used by the microservice.
    - **what for**: Provides a more accurate picture of the actual memory usage and can be used to identify memory-related performance issues or to optimize memory usage.
    - **dashboard**: -//-

### Real-time Metrics:
- `prediction_price_histogram`
    - **type**: Histogram
    - **description**: Used to build a histogram of the prediction prices.
    - **what for**: Can help to understand the distribution of the prediction prices and can be used to identify anomalies by checking different quantiles.
    - **dashboard**: Used in the `Price Prediction Quantiles (total)` and `Price Prediction Quantiles (over time window)` time series panels, where the distribution of the prediction prices is shown for 0.05, 0.5 and 0.95 quantiles.
- `http_request_duration_seconds`
    - **type**: Histogram
    - **description**: Used to build a histogram of the time it takes to process the requests (delay). Calculated separately for valid and invalid requests.
    - **what for**: Can help to understand the distribution of the request processing time and can be used to identify anomalies by checking different quantiles.
    - **dashboard**: Used in the `Prediction Request Duration Quantiles (total)` and `Prediction Request Duration Quantiles (over time window)` time series panels, where the distribution of the request processing time is shown for 0.05, 0.5 and 0.95 quantiles. Also used in the `Prediction Request Rate` time series panel.

### Application Layer Metrics:
- `invalid_request_count`
    - **type**: Counter
    - **description**: Measures the number of invalid requests received by the microservice.
    - **what for**: Can help to verify if the users understand the microservice's use cases especially in terms of the range of model's paremeters (e.g. model can't be used if `total_area`>100).
    - **dashboard**: Used in the `Error Rate` and `Error Increase` time seriees panels. This way the user can see the dynamics in a change of the rate or in the absolute value of the error.
- `failed_predictions`
    - **type**: Counter
    - **description**: Measures the number of failed predictions due to the model's prediction errors.
    - **what for**: Can help to identify if the model works well as it should be since it can only receive a valid request. In case of increase of this metric, the developer should check if the model works properly.
    - **dashboard**: -//-
- `too_many_requests_count_total`
    - **type**: Counter
    - **description**: Measures the number of requests received by the microservice that were rejected due to the rate limiter.
    - **what for**: Can help to check if the microservice is under heavy load (in which case the rate limiter should be increased possibly along with the hardware's resources)
    - **dashboard**: -//-
- `unexpected_error_count_total`
    - **type**: Counter
    - **description**: Measures the number of unexpected errors that occurred during the processing of the requests by the microservice.
    - **what for**: Can help to identify if the microservice is working properly and if it has any bugs.
    - **dashboard**: -//-

