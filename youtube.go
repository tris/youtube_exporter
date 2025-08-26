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
	// Now also fetching contentDetails to get the uploads playlist ID
	call := ytSvc.Channels.List([]string{"snippet", "statistics", "contentDetails"})
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

// fetchChannelLiveStreams fetches live streams from a channel using the EFFICIENT method
// This uses the uploads playlist instead of Search API, saving ~97 quota units!
// Old method: 100 units (Search.List) + 1 unit (Videos.List) = 101 units
// New method: 2 units (PlaylistItems.List x2) + 2 units (Videos.List x2) = 4 units
func fetchChannelLiveStreams(ctx context.Context, ytSvc *youtube.Service, channel *youtube.Channel) ([]*youtube.Video, error) {
	// Get uploads playlist ID from channel contentDetails
	if channel.ContentDetails == nil || channel.ContentDetails.RelatedPlaylists == nil {
		return nil, nil
	}

	uploadsPlaylistID := channel.ContentDetails.RelatedPlaylists.Uploads
	if uploadsPlaylistID == "" {
		return nil, nil
	}

	// Fetch recent videos from uploads playlist (1 unit per 50 videos)
	// We'll check the 100 most recent videos (2 units total)
	var allVideoIDs []string
	nextPageToken := ""
	pagesChecked := 0
	maxPages := 2 // Check up to 100 videos (2 pages of 50)

	for pagesChecked < maxPages {
		playlistCall := ytSvc.PlaylistItems.List([]string{"contentDetails"})
		playlistCall = playlistCall.PlaylistId(uploadsPlaylistID)
		playlistCall = playlistCall.MaxResults(50)
		if nextPageToken != "" {
			playlistCall = playlistCall.PageToken(nextPageToken)
		}
		playlistCall.Context(ctx)

		playlistResp, err := playlistCall.Do()
		if err != nil {
			log.Printf("Error fetching playlist items: %v", err)
			break
		}

		for _, item := range playlistResp.Items {
			if item.ContentDetails != nil && item.ContentDetails.VideoId != "" {
				allVideoIDs = append(allVideoIDs, item.ContentDetails.VideoId)
			}
		}

		nextPageToken = playlistResp.NextPageToken
		pagesChecked++

		if nextPageToken == "" || len(playlistResp.Items) == 0 {
			break
		}
	}

	if len(allVideoIDs) == 0 {
		return nil, nil
	}

	// Fetch video details in batches of 50 (1 unit per batch)
	var allLiveVideos []*youtube.Video

	for i := 0; i < len(allVideoIDs); i += 50 {
		end := i + 50
		if end > len(allVideoIDs) {
			end = len(allVideoIDs)
		}

		batchIDs := allVideoIDs[i:end]

		videosCall := ytSvc.Videos.List([]string{"snippet", "statistics", "liveStreamingDetails"})
		videosCall = videosCall.Id(batchIDs...)
		videosCall.Context(ctx)

		videosResp, err := videosCall.Do()
		if err != nil {
			log.Printf("Error fetching video details: %v", err)
			continue
		}

		// Filter for only live videos
		for _, video := range videosResp.Items {
			if video.Snippet != nil && video.Snippet.LiveBroadcastContent == "live" {
				allLiveVideos = append(allLiveVideos, video)
			}
		}
	}

	log.Printf("Found %d live streams using efficient method (checked %d recent videos)", len(allLiveVideos), len(allVideoIDs))
	return allLiveVideos, nil
}

// processChannelData processes YouTube API response into a channel snapshot
// UPDATED to use the efficient live stream fetching method
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

	// Fetch live streams using the EFFICIENT method
	// Pass the channel object which already has contentDetails
	liveVideos, err := fetchChannelLiveStreams(ctx, ytSvc, channel)
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
