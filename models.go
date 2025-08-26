package main

import (
	"regexp"
)

// Simple video ID check: standard IDs are 11 chars of URL-safe base64-ish set.
var videoIDRe = regexp.MustCompile(`^[A-Za-z0-9_-]{11}$`)

// data snapshot for one scrape
type ytSnapshot struct {
	videoID            string
	title              string
	viewCount          float64
	likeCount          float64
	concurrentViewers  float64
	liveBroadcastState string  // "live" | "upcoming" | "none" (per snippet)
	liveBinary         float64 // 1 if live, else 0
}
