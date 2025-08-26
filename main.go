package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/api/option"
	"google.golang.org/api/youtube/v3"
)

var (
	addrFlag   = flag.String("addr", ":9473", "listen address")
	apiKeyFlag = flag.String("api-key", "", "YouTube Data API key (overrides YOUTUBE_API_KEY)")
)

// Simple video ID check: standard IDs are 11 chars of URL-safe base64-ish set.
var videoIDRe = regexp.MustCompile(`^[A-Za-z0-9_-]{11}$`)

// data snapshot for one scrape
type ytSnapshot struct {
	videoID            string
	viewCount          float64
	likeCount          float64
	concurrentViewers  float64
	liveBroadcastState string // "live" | "upcoming" | "none" (per snippet)
	liveBinary         float64 // 1 if live, else 0
}

type snapshotCollector struct {
	s             ytSnapshot
	viewDesc      *prometheus.Desc
	likeDesc      *prometheus.Desc
	concurDesc    *prometheus.Desc
	liveDesc      *prometheus.Desc
	liveStateDesc *prometheus.Desc
	upDesc        *prometheus.Desc
}

func newSnapshotCollector(s ytSnapshot) *snapshotCollector {
	labels := []string{"video_id"}
	return &snapshotCollector{
		s: s,
		viewDesc: prometheus.NewDesc(
			"youtube_video_view_count",
			"Total view count reported by YouTube for this video.",
			labels, nil),
		likeDesc: prometheus.NewDesc(
			"youtube_video_like_count",
			"Total like count reported by YouTube for this video.",
			labels, nil),
		concurDesc: prometheus.NewDesc(
			"youtube_video_concurrent_viewers",
			"Concurrent viewers (only non-zero while live) reported by YouTube for this video.",
			labels, nil),
		liveDesc: prometheus.NewDesc(
			"youtube_video_live",
			"1 if YouTube reports the video as currently live, else 0.",
			labels, nil),
		liveStateDesc: prometheus.NewDesc(
			"youtube_video_live_status",
			"Infometric with state label; 1 for the current state (\"live\", \"upcoming\", or \"none\").",
			[]string{"video_id", "state"}, nil),
		upDesc: prometheus.NewDesc(
			"youtube_video_scrape_success",
			"1 if the scrape of YouTube API succeeded, else 0.",
			labels, nil),
	}
}

func (c *snapshotCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.viewDesc
	ch <- c.likeDesc
	ch <- c.concurDesc
	ch <- c.liveDesc
	ch <- c.liveStateDesc
	ch <- c.upDesc
}

func (c *snapshotCollector) Collect(ch chan<- prometheus.Metric) {
	lbls := []string{c.s.videoID}
	ch <- prometheus.MustNewConstMetric(c.viewDesc, prometheus.GaugeValue, c.s.viewCount, lbls...)
	ch <- prometheus.MustNewConstMetric(c.likeDesc, prometheus.GaugeValue, c.s.likeCount, lbls...)
	ch <- prometheus.MustNewConstMetric(c.concurDesc, prometheus.GaugeValue, c.s.concurrentViewers, lbls...)
	ch <- prometheus.MustNewConstMetric(c.liveDesc, prometheus.GaugeValue, c.s.liveBinary, lbls...)
	// infometric: 1 for the current state only
	ch <- prometheus.MustNewConstMetric(c.liveStateDesc, prometheus.GaugeValue, 1, c.s.videoID, c.s.liveBroadcastState)
	ch <- prometheus.MustNewConstMetric(c.upDesc, prometheus.GaugeValue, 1, lbls...)
}

func main() {
	flag.Parse()

	apiKey := *apiKeyFlag
	if apiKey == "" {
		apiKey = os.Getenv("YOUTUBE_API_KEY")
	}
	if apiKey == "" {
		log.Fatal("You must provide an API key via -api-key or YOUTUBE_API_KEY")
	}

	ctx := context.Background()
	ytSvc, err := youtube.NewService(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("youtube.NewService: %v", err)
	}

	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
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

		call := ytSvc.Videos.List([]string{"snippet", "statistics", "liveStreamingDetails"})
		call = call.Id(videoID)
		// Ensure the HTTP request respects our per-request context
		call.Context(reqCtx)

		resp, err := call.Do()
		if err != nil {
			http.Error(w, fmt.Sprintf("YouTube API error: %v", err), http.StatusBadGateway)
			return
		}
		if len(resp.Items) == 0 {
			http.Error(w, "video not found", http.StatusNotFound)
			return
		}
		it := resp.Items[0]

		stats := it.Statistics
		lsd := it.LiveStreamingDetails

		// The Go client exposes numeric counts as uint64 with ,string JSON tags.
		var viewCount float64
		var likeCount float64
		var concurrent float64

		if stats != nil {
			viewCount = float64(stats.ViewCount)
			likeCount = float64(stats.LikeCount)
		}
		if lsd != nil {
			concurrent = float64(lsd.ConcurrentViewers)
		}

		state := "none"
		if it.Snippet != nil && it.Snippet.LiveBroadcastContent != "" {
			state = it.Snippet.LiveBroadcastContent // "live" | "upcoming" | "none"
		}
		liveBinary := 0.0
		if state == "live" {
			liveBinary = 1.0
		}

		snap := ytSnapshot{
			videoID:            videoID,
			viewCount:          viewCount,
			likeCount:          likeCount,
			concurrentViewers:  concurrent,
			liveBroadcastState: state,
			liveBinary:         liveBinary,
		}

		// Fresh, per-request registry that only contains THIS video's metrics.
		reg := prometheus.NewRegistry()
		reg.MustRegister(newSnapshotCollector(snap))

		// Serve metrics for just this registry.
		promhttp.HandlerFor(reg, promhttp.HandlerOpts{
			// Best-effort: surface any internal collection errors as 500s rather than swallowing them.
			ErrorHandling: promhttp.HTTPErrorOnError,
		}).ServeHTTP(w, r)
	})

	log.Printf("Listening on %s …", *addrFlag)
	if err := http.ListenAndServe(*addrFlag, nil); err != nil {
		log.Fatal(err)
	}
}
