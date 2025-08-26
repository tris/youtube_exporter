package main

import (
	"github.com/prometheus/client_golang/prometheus"
)

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
	labels := []string{"video_id", "title"}
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
			[]string{"video_id", "title", "state"}, nil),
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
	lbls := []string{c.s.videoID, c.s.title}
	ch <- prometheus.MustNewConstMetric(c.viewDesc, prometheus.GaugeValue, c.s.viewCount, lbls...)
	ch <- prometheus.MustNewConstMetric(c.likeDesc, prometheus.GaugeValue, c.s.likeCount, lbls...)
	ch <- prometheus.MustNewConstMetric(c.concurDesc, prometheus.GaugeValue, c.s.concurrentViewers, lbls...)
	ch <- prometheus.MustNewConstMetric(c.liveDesc, prometheus.GaugeValue, c.s.liveBinary, lbls...)
	// infometric: 1 for the current state only
	ch <- prometheus.MustNewConstMetric(c.liveStateDesc, prometheus.GaugeValue, 1, c.s.videoID, c.s.title, c.s.liveBroadcastState)
	ch <- prometheus.MustNewConstMetric(c.upDesc, prometheus.GaugeValue, 1, lbls...)
}
