package main

import (
	"context"
	"log"

	"google.golang.org/api/option"
	"google.golang.org/api/youtube/v3"
)

// newYouTubeService creates a new YouTube service client
func newYouTubeService(ctx context.Context, apiKey string) (*youtube.Service, error) {
	ytSvc, err := youtube.NewService(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Printf("youtube.NewService: %v", err)
		return nil, err
	}
	return ytSvc, nil
}

// fetchVideoDetails fetches video details from YouTube API
func fetchVideoDetails(ctx context.Context, ytSvc *youtube.Service, videoID string) (*youtube.Video, error) {
	call := ytSvc.Videos.List([]string{"snippet", "statistics", "liveStreamingDetails"})
	call = call.Id(videoID)
	call.Context(ctx)

	resp, err := call.Do()
	if err != nil {
		return nil, err
	}

	if len(resp.Items) == 0 {
		return nil, nil // Video not found
	}

	return resp.Items[0], nil
}

// processVideoData processes YouTube API response into a snapshot
func processVideoData(video *youtube.Video, videoID string) ytSnapshot {
	stats := video.Statistics
	lsd := video.LiveStreamingDetails

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
	if video.Snippet != nil && video.Snippet.LiveBroadcastContent != "" {
		state = video.Snippet.LiveBroadcastContent // "live" | "upcoming" | "none"
	}
	liveBinary := 0.0
	if state == "live" {
		liveBinary = 1.0
	}

	// Extract title from snippet
	var title string
	if video.Snippet != nil {
		title = video.Snippet.Title
	}

	return ytSnapshot{
		videoID:            videoID,
		title:              title,
		viewCount:          viewCount,
		likeCount:          likeCount,
		concurrentViewers:  concurrent,
		liveBroadcastState: state,
		liveBinary:         liveBinary,
	}
}
