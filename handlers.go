package main

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/api/youtube/v3"
)

// metricsHandler handles the /metrics endpoint
func metricsHandler(ytSvc *youtube.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		videoID := r.URL.Query().Get("v")
		if videoID == "" {
			http.Error(w, "missing required query parameter: v (YouTube video ID)", http.StatusBadRequest)
			return
		}
		// keep exporter well-behaved; reject clearly bad IDs
		if !videoIDRe.MatchString(videoID) {
			http.Error(w, "invalid video ID format", http.StatusBadRequest)
			return
		}

		// Query YouTube API on *each* scrape request (no background work).
		reqCtx, cancel := context.WithTimeout(r.Context(), 8*time.Second)
		defer cancel()

		video, err := fetchVideoDetails(reqCtx, ytSvc, videoID)
		if err != nil {
			http.Error(w, fmt.Sprintf("YouTube API error: %v", err), http.StatusBadGateway)
			return
		}
		if video == nil {
			http.Error(w, "video not found", http.StatusNotFound)
			return
		}

		snap := processVideoData(video, videoID)

		// Fresh, per-request registry that only contains THIS video's metrics.
		reg := prometheus.NewRegistry()
		reg.MustRegister(newSnapshotCollector(snap))

		// Serve metrics for just this registry.
		promhttp.HandlerFor(reg, promhttp.HandlerOpts{
			// Best-effort: surface any internal collection errors as 500s rather than swallowing them.
			ErrorHandling: promhttp.HTTPErrorOnError,
		}).ServeHTTP(w, r)
	}
}
