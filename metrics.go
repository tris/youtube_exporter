package main

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"google.golang.org/api/googleapi"
)

var (
	// Counts estimated YouTube Data API quota units consumed, labeled by endpoint.
	apiQuotaUnits = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "youtube_api_quota_units_total",
			Help: "Estimated YouTube Data API quota units consumed, labeled by endpoint.",
		},
		[]string{"endpoint"},
	)

	// Counts API request errors, labeled by endpoint and HTTP status code (if available).
	apiErrors = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "youtube_api_errors_total",
			Help: "YouTube Data API request errors, labeled by endpoint and status code.",
		},
		[]string{"endpoint", "code"},
	)
)

func init() {
	prometheus.MustRegister(apiQuotaUnits)
	prometheus.MustRegister(apiErrors)
}

// addQuotaUnits increments the quota unit counter for an endpoint.
func addQuotaUnits(endpoint string, units float64) {
	apiQuotaUnits.WithLabelValues(endpoint).Add(units)
}

// incAPIError increments the error counter for an endpoint with status code extracted from error.
func incAPIError(endpoint string, err error) {
	code := codeFromError(err)
	apiErrors.WithLabelValues(endpoint, code).Inc()
}

func codeFromError(err error) string {
	if err == nil {
		return "0"
	}
	if ge, ok := err.(*googleapi.Error); ok {
		return strconv.Itoa(ge.Code)
	}
	return "unknown"
}
