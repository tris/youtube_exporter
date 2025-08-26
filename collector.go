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

type channelSnapshotCollector struct {
	s              ytChannelSnapshot
	subscriberDesc *prometheus.Desc
	viewDesc       *prometheus.Desc
	videoDesc      *prometheus.Desc
	upDesc         *prometheus.Desc
	// Live stream metrics (using same names as individual video metrics)
	liveStreamViewDesc   *prometheus.Desc
	liveStreamLikeDesc   *prometheus.Desc
	liveStreamConcurDesc *prometheus.Desc
	liveStreamLiveDesc   *prometheus.Desc
	liveStreamStateDesc  *prometheus.Desc
	liveStreamUpDesc     *prometheus.Desc
}

func newChannelSnapshotCollector(s ytChannelSnapshot) *channelSnapshotCollector {
	channelLabels := []string{"channel_id", "channel_title"}
	videoLabels := []string{"video_id", "title"}

	return &channelSnapshotCollector{
		s: s,
		subscriberDesc: prometheus.NewDesc(
			"youtube_channel_subscriber_count",
			"Total subscriber count reported by YouTube for this channel.",
			channelLabels, nil),
		viewDesc: prometheus.NewDesc(
			"youtube_channel_view_count",
			"Total view count reported by YouTube for this channel.",
			channelLabels, nil),
		videoDesc: prometheus.NewDesc(
			"youtube_channel_video_count",
			"Total video count reported by YouTube for this channel.",
			channelLabels, nil),
		upDesc: prometheus.NewDesc(
			"youtube_channel_scrape_success",
			"1 if the scrape of YouTube API succeeded, else 0.",
			channelLabels, nil),
		// Live stream metrics (using same metric names as individual video queries)
		liveStreamViewDesc: prometheus.NewDesc(
			"youtube_video_view_count",
			"Total view count reported by YouTube for this video.",
			videoLabels, nil),
		liveStreamLikeDesc: prometheus.NewDesc(
			"youtube_video_like_count",
			"Total like count reported by YouTube for this video.",
			videoLabels, nil),
		liveStreamConcurDesc: prometheus.NewDesc(
			"youtube_video_concurrent_viewers",
			"Concurrent viewers (only non-zero while live) reported by YouTube for this video.",
			videoLabels, nil),
		liveStreamLiveDesc: prometheus.NewDesc(
			"youtube_video_live",
			"1 if YouTube reports the video as currently live, else 0.",
			videoLabels, nil),
		liveStreamStateDesc: prometheus.NewDesc(
			"youtube_video_live_status",
			"Infometric with state label; 1 for the current state (\"live\", \"upcoming\", or \"none\").",
			[]string{"video_id", "title", "state"}, nil),
		liveStreamUpDesc: prometheus.NewDesc(
			"youtube_video_scrape_success",
			"1 if the scrape of YouTube API succeeded, else 0.",
			videoLabels, nil),
	}
}

func (c *channelSnapshotCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.subscriberDesc
	ch <- c.viewDesc
	ch <- c.videoDesc
	ch <- c.upDesc
	ch <- c.liveStreamViewDesc
	ch <- c.liveStreamLikeDesc
	ch <- c.liveStreamConcurDesc
	ch <- c.liveStreamLiveDesc
	ch <- c.liveStreamStateDesc
	ch <- c.liveStreamUpDesc
}

func (c *channelSnapshotCollector) Collect(ch chan<- prometheus.Metric) {
	channelLbls := []string{c.s.channelID, c.s.title}

	// Channel-level metrics
	ch <- prometheus.MustNewConstMetric(c.subscriberDesc, prometheus.GaugeValue, c.s.subscriberCount, channelLbls...)
	ch <- prometheus.MustNewConstMetric(c.viewDesc, prometheus.GaugeValue, c.s.viewCount, channelLbls...)
	ch <- prometheus.MustNewConstMetric(c.videoDesc, prometheus.GaugeValue, c.s.videoCount, channelLbls...)
	ch <- prometheus.MustNewConstMetric(c.upDesc, prometheus.GaugeValue, 1, channelLbls...)

	// Live stream metrics (using same metric names as individual video queries)
	for _, stream := range c.s.liveStreams {
		videoLbls := []string{stream.videoID, stream.title}

		ch <- prometheus.MustNewConstMetric(c.liveStreamViewDesc, prometheus.GaugeValue, stream.viewCount, videoLbls...)
		ch <- prometheus.MustNewConstMetric(c.liveStreamLikeDesc, prometheus.GaugeValue, stream.likeCount, videoLbls...)
		ch <- prometheus.MustNewConstMetric(c.liveStreamConcurDesc, prometheus.GaugeValue, stream.concurrentViewers, videoLbls...)
		ch <- prometheus.MustNewConstMetric(c.liveStreamLiveDesc, prometheus.GaugeValue, stream.liveBinary, videoLbls...)

		// Live stream state infometric
		stateLbls := []string{stream.videoID, stream.title, stream.liveBroadcastState}
		ch <- prometheus.MustNewConstMetric(c.liveStreamStateDesc, prometheus.GaugeValue, 1, stateLbls...)

		ch <- prometheus.MustNewConstMetric(c.liveStreamUpDesc, prometheus.GaugeValue, 1, videoLbls...)
	}
}
