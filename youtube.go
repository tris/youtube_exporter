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

// fetchChannelDetails fetches channel details from YouTube API
func fetchChannelDetails(ctx context.Context, ytSvc *youtube.Service, channelID string) (*youtube.Channel, error) {
	call := ytSvc.Channels.List([]string{"snippet", "statistics"})
	call = call.Id(channelID)
	call.Context(ctx)

	resp, err := call.Do()
	if err != nil {
		return nil, err
	}

	if len(resp.Items) == 0 {
		return nil, nil // Channel not found
	}

	return resp.Items[0], nil
}

// fetchChannelLiveStreams fetches all live streams from a channel
func fetchChannelLiveStreams(ctx context.Context, ytSvc *youtube.Service, channelID string) ([]*youtube.Video, error) {
	var allVideos []*youtube.Video
	nextPageToken := ""

	for {
		// Search for live videos from this channel
		searchCall := ytSvc.Search.List([]string{"id"})
		searchCall = searchCall.ChannelId(channelID)
		searchCall = searchCall.EventType("live")
		searchCall = searchCall.Type("video")
		searchCall = searchCall.MaxResults(50)
		if nextPageToken != "" {
			searchCall = searchCall.PageToken(nextPageToken)
		}
		searchCall.Context(ctx)

		searchResp, err := searchCall.Do()
		if err != nil {
			return nil, err
		}

		if len(searchResp.Items) == 0 {
			break
		}

		// Extract video IDs
		var videoIDs []string
		for _, item := range searchResp.Items {
			if item.Id != nil && item.Id.VideoId != "" {
				videoIDs = append(videoIDs, item.Id.VideoId)
			}
		}

		if len(videoIDs) == 0 {
			break
		}

		// Fetch detailed video information
		videosCall := ytSvc.Videos.List([]string{"snippet", "statistics", "liveStreamingDetails"})
		videosCall = videosCall.Id(videoIDs...)
		videosCall.Context(ctx)

		videosResp, err := videosCall.Do()
		if err != nil {
			return nil, err
		}

		allVideos = append(allVideos, videosResp.Items...)

		// Check if there are more pages
		nextPageToken = searchResp.NextPageToken
		if nextPageToken == "" {
			break
		}
	}

	return allVideos, nil
}

// processChannelData processes YouTube API response into a channel snapshot
func processChannelData(ctx context.Context, ytSvc *youtube.Service, channel *youtube.Channel, channelID string) (ytChannelSnapshot, error) {
	stats := channel.Statistics

	var subscriberCount float64
	var viewCount float64
	var videoCount float64

	if stats != nil {
		subscriberCount = float64(stats.SubscriberCount)
		viewCount = float64(stats.ViewCount)
		videoCount = float64(stats.VideoCount)
	}

	// Extract title from snippet
	var title string
	if channel.Snippet != nil {
		title = channel.Snippet.Title
	}

	// Fetch live streams for this channel
	liveVideos, err := fetchChannelLiveStreams(ctx, ytSvc, channelID)
	if err != nil {
		log.Printf("Error fetching live streams for channel %s: %v", channelID, err)
		// Continue without live streams rather than failing completely
		liveVideos = []*youtube.Video{}
	}

	// Process each live stream
	var liveStreams []ytSnapshot
	for _, video := range liveVideos {
		snapshot := processVideoData(video, video.Id)
		liveStreams = append(liveStreams, snapshot)
	}

	return ytChannelSnapshot{
		channelID:       channelID,
		title:           title,
		subscriberCount: subscriberCount,
		viewCount:       viewCount,
		videoCount:      videoCount,
		liveStreams:     liveStreams,
	}, nil
}
