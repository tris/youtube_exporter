package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/api/youtube/v3"
)

// metricsHandler handles the /scrape endpoint
func metricsHandler(ytSvc *youtube.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		videoID := r.URL.Query().Get("v")
		channelID := r.URL.Query().Get("channel")
		disableLive := r.URL.Query().Get("disable_live") == "true"

		// Check if both or neither parameters are provided
		if videoID == "" && channelID == "" {
			http.Error(w, "missing required query parameter: v (YouTube video ID) or channel (YouTube channel ID)", http.StatusBadRequest)
			return
		}
		if videoID != "" && channelID != "" {
			http.Error(w, "provide either v (video ID) or channel (channel ID), not both", http.StatusBadRequest)
			return
		}

		// Query YouTube API on *each* scrape request (no background work).
		reqCtx, cancel := context.WithTimeout(r.Context(), 30*time.Second) // Increased timeout for channel requests
		defer cancel()

		// Fresh, per-request registry
		reg := prometheus.NewRegistry()

		if videoID != "" {
			// Handle video metrics (existing functionality)
			if !videoIDRe.MatchString(videoID) {
				http.Error(w, "invalid video ID format", http.StatusBadRequest)
				return
			}

			video, err := fetchVideoDetails(reqCtx, ytSvc, videoID)
			if err != nil {
				log.Printf("metricsHandler: Failed to fetch details for video %s: %v", videoID, err)
				http.Error(w, fmt.Sprintf("YouTube API error: %v", err), http.StatusBadGateway)
				return
			}
			if video == nil {
				http.Error(w, "video not found", http.StatusNotFound)
				return
			}

			snap := processVideoData(video, videoID)
			reg.MustRegister(newSnapshotCollector(snap))

		} else if channelID != "" {
			// Handle channel metrics (new functionality)
			if !channelIDRe.MatchString(channelID) {
				http.Error(w, "invalid channel ID format", http.StatusBadRequest)
				return
			}

			channel, err := fetchChannelDetails(reqCtx, ytSvc, channelID)
			if err != nil {
				log.Printf("metricsHandler: Failed to fetch details for channel %s: %v", channelID, err)
				http.Error(w, fmt.Sprintf("YouTube API error: %v", err), http.StatusBadGateway)
				return
			}
			if channel == nil {
				http.Error(w, "channel not found", http.StatusNotFound)
				return
			}

			channelSnap, err := processChannelData(reqCtx, ytSvc, channel, channelID, disableLive)
			if err != nil {
				log.Printf("metricsHandler: Failed to process data for channel %s (%s): %v", channelID, channel.Snippet.Title, err)
				http.Error(w, fmt.Sprintf("YouTube API error processing channel data: %v", err), http.StatusBadGateway)
				return
			}

			reg.MustRegister(newChannelSnapshotCollector(channelSnap))
		}

		// Serve metrics for just this registry.
		promhttp.HandlerFor(reg, promhttp.HandlerOpts{
			// Best-effort: surface any internal collection errors as 500s rather than swallowing them.
			ErrorHandling: promhttp.HTTPErrorOnError,
		}).ServeHTTP(w, r)
	}
}
