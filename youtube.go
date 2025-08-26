package main

import (
	"context"
	"log"
	"strings"
	"sync"
	"time"

	"google.golang.org/api/option"
	"google.golang.org/api/youtube/v3"
)

// LiveStreamCache stores cached live stream IDs for channels
type LiveStreamCache struct {
	mu    sync.RWMutex
	cache map[string]*ChannelLiveCache
}

// ChannelLiveCache stores live stream data for a specific channel
type ChannelLiveCache struct {
	Initialized   bool
	CachedLiveIDs map[string]struct{}
	LastUpdated   time.Time
}

// Global cache instance
var liveStreamCache = &LiveStreamCache{
	cache: make(map[string]*ChannelLiveCache),
}

// GetChannelCache retrieves or creates cache entry for a channel
func (c *LiveStreamCache) GetChannelCache(channelID string) *ChannelLiveCache {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.cache[channelID]; !exists {
		c.cache[channelID] = &ChannelLiveCache{
			Initialized:   false,
			CachedLiveIDs: make(map[string]struct{}),
		}
	}
	return c.cache[channelID]
}

// UpdateCache updates the cached live IDs for a channel
func (c *LiveStreamCache) UpdateCache(channelID string, liveIDs map[string]struct{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if channelCache, exists := c.cache[channelID]; exists {
		channelCache.CachedLiveIDs = liveIDs
		channelCache.LastUpdated = time.Now()
		channelCache.Initialized = true
	}
}

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

// fetchChannelLiveStreamsComprehensive uses the expensive Search API to find ALL live streams
func fetchChannelLiveStreamsComprehensive(ctx context.Context, ytSvc *youtube.Service, channelID string) ([]*youtube.Video, error) {
	log.Printf("Performing comprehensive live stream search for channel %s (100 units)", channelID)

	// Use the Search API to find all live streams for this channel
	searchCall := ytSvc.Search.List([]string{"id"}).
		ChannelId(channelID).
		EventType("live").
		Type("video").
		MaxResults(50)

	searchResp, err := searchCall.Context(ctx).Do()
	if err != nil {
		return nil, err
	}

	if len(searchResp.Items) == 0 {
		return []*youtube.Video{}, nil
	}

	// Extract video IDs
	var videoIDs []string
	for _, item := range searchResp.Items {
		if item.Id != nil && item.Id.VideoId != "" {
			videoIDs = append(videoIDs, item.Id.VideoId)
		}
	}

	// Fetch full video details (batched)
	videosCall := ytSvc.Videos.List([]string{"snippet", "statistics", "liveStreamingDetails"})
	videosCall = videosCall.Id(strings.Join(videoIDs, ","))
	videosCall.Context(ctx)

	videosResp, err := videosCall.Do()
	if err != nil {
		return nil, err
	}

	return videosResp.Items, nil
}

// fetchAllChannelVideos fetches ALL videos from a channel's uploads playlist (for full pagination)
func fetchAllChannelVideos(ctx context.Context, ytSvc *youtube.Service, channel *youtube.Channel) ([]*youtube.Video, error) {
	// Get the uploads playlist ID from channel's contentDetails
	if channel.ContentDetails == nil || channel.ContentDetails.RelatedPlaylists == nil {
		return nil, nil
	}

	uploadsPlaylistID := channel.ContentDetails.RelatedPlaylists.Uploads
	if uploadsPlaylistID == "" {
		return nil, nil
	}

	// Fetch ALL videos from the uploads playlist
	var allVideoIDs []string
	nextPageToken := ""
	pagesChecked := 0
	maxPages := 99 // Up to 4950 videos (99 pages of 50)

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
		return []*youtube.Video{}, nil
	}

	log.Printf("Fetched %d video IDs from channel %s (%d pages)", len(allVideoIDs), channel.Id, pagesChecked)

	// Fetch video details in batches of 50 (1 unit per batch)
	var allLiveVideos []*youtube.Video

	for i := 0; i < len(allVideoIDs); i += 50 {
		end := i + 50
		if end > len(allVideoIDs) {
			end = len(allVideoIDs)
		}

		batchIDs := allVideoIDs[i:end]

		videosCall := ytSvc.Videos.List([]string{"snippet", "statistics", "liveStreamingDetails"})
		videosCall = videosCall.Id(strings.Join(batchIDs, ","))
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

	return allLiveVideos, nil
}

// fetchRecentLiveStreams fetches recent live streams using the playlist method
func fetchRecentLiveStreams(ctx context.Context, ytSvc *youtube.Service, channel *youtube.Channel) ([]*youtube.Video, error) {
	// Get the uploads playlist ID from channel's contentDetails
	if channel.ContentDetails == nil || channel.ContentDetails.RelatedPlaylists == nil {
		return nil, nil
	}

	uploadsPlaylistID := channel.ContentDetails.RelatedPlaylists.Uploads
	if uploadsPlaylistID == "" {
		return nil, nil
	}

	// Fetch recent videos from the uploads playlist
	// For subsequent requests, 50 videos (1 page) should be sufficient since we poll every minute
	var allVideoIDs []string
	nextPageToken := ""
	pagesChecked := 0
	maxPages := 1 // Check up to 50 videos (1 page) - sufficient for minute-interval polling

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
		return []*youtube.Video{}, nil
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
		videosCall = videosCall.Id(strings.Join(batchIDs, ","))
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

	return allLiveVideos, nil
}

// refreshCachedVideos checks if cached video IDs are still live
func refreshCachedVideos(ctx context.Context, ytSvc *youtube.Service, cachedIDs map[string]struct{}) ([]*youtube.Video, map[string]struct{}) {
	if len(cachedIDs) == 0 {
		return []*youtube.Video{}, make(map[string]struct{})
	}

	// Convert map to slice for API call
	var idSlice []string
	for id := range cachedIDs {
		idSlice = append(idSlice, id)
	}

	var stillLiveVideos []*youtube.Video
	stillLiveIDs := make(map[string]struct{})

	// Fetch in batches of 50
	for i := 0; i < len(idSlice); i += 50 {
		end := i + 50
		if end > len(idSlice) {
			end = len(idSlice)
		}

		batch := idSlice[i:end]
		videosCall := ytSvc.Videos.List([]string{"snippet", "statistics", "liveStreamingDetails"})
		videosCall = videosCall.Id(strings.Join(batch, ","))
		videosCall.Context(ctx)

		videosResp, err := videosCall.Do()
		if err != nil {
			log.Printf("Error refreshing cached videos: %v", err)
			continue
		}

		// Check which are still live
		for _, video := range videosResp.Items {
			if video.Snippet != nil && video.Snippet.LiveBroadcastContent == "live" {
				stillLiveVideos = append(stillLiveVideos, video)
				stillLiveIDs[video.Id] = struct{}{}
			}
		}
	}

	return stillLiveVideos, stillLiveIDs
}

// fetchChannelLiveStreams fetches live streams with caching for long-running streams
func fetchChannelLiveStreams(ctx context.Context, ytSvc *youtube.Service, channel *youtube.Channel, disableLive bool) ([]*youtube.Video, error) {
	// If live stream fetching is disabled, return empty
	if disableLive {
		log.Printf("Live stream fetching disabled for channel %s", channel.Id)
		return []*youtube.Video{}, nil
	}

	channelCache := liveStreamCache.GetChannelCache(channel.Id)

	// Determine the best strategy based on channel size
	var videoCount uint64
	if channel.Statistics != nil {
		videoCount = channel.Statistics.VideoCount
	}

	// Strategy selection:
	// - ≤4950 videos (99 pages): Use full pagination (cheaper than Search API)
	// - >4950 videos: Use Search API for comprehensive search
	useFullPagination := videoCount <= 4950

	// If this is the first time, decide between full pagination or Search API
	if !channelCache.Initialized {
		if useFullPagination {
			// For channels with ≤4950 videos, paginate through all videos
			log.Printf("Channel %s has %d videos (≤4950), using full pagination instead of Search API", channel.Id, videoCount)

			allVideos, err := fetchAllChannelVideos(ctx, ytSvc, channel)
			if err != nil {
				log.Printf("Error in full pagination: %v", err)
				// Continue with recent videos check
			} else {
				// Cache the live video IDs
				newCache := make(map[string]struct{})
				for _, video := range allVideos {
					newCache[video.Id] = struct{}{}
				}
				liveStreamCache.UpdateCache(channel.Id, newCache)
				log.Printf("Found %d live streams via full pagination for channel %s", len(newCache), channel.Id)

				// Return the live videos directly since we just fetched them
				return allVideos, nil
			}
		} else {
			// For large channels (>4950 videos), use Search API
			log.Printf("Channel %s has %d videos (>4950), using Search API", channel.Id, videoCount)

			comprehensiveVideos, err := fetchChannelLiveStreamsComprehensive(ctx, ytSvc, channel.Id)
			if err != nil {
				log.Printf("Error in comprehensive search: %v", err)
				// Continue with playlist method even if comprehensive search fails
			} else {
				// Cache the live video IDs
				newCache := make(map[string]struct{})
				for _, video := range comprehensiveVideos {
					newCache[video.Id] = struct{}{}
				}
				liveStreamCache.UpdateCache(channel.Id, newCache)
				log.Printf("Cached %d live streams from comprehensive search for channel %s", len(newCache), channel.Id)
			}
		}
	}

	// Always fetch recent videos using the efficient playlist method
	recentLiveVideos, err := fetchRecentLiveStreams(ctx, ytSvc, channel)
	if err != nil {
		return nil, err
	}

	// Create a map to track all live videos (deduplication)
	allLiveVideos := make(map[string]*youtube.Video)

	// Add recent live videos
	for _, video := range recentLiveVideos {
		allLiveVideos[video.Id] = video
	}

	// If we have cached IDs, refresh their status
	if channelCache.Initialized && len(channelCache.CachedLiveIDs) > 0 {
		cachedVideos, stillLiveIDs := refreshCachedVideos(ctx, ytSvc, channelCache.CachedLiveIDs)

		// Add still-live cached videos
		for _, video := range cachedVideos {
			allLiveVideos[video.Id] = video
		}

		// Update cache with only still-live IDs
		liveStreamCache.UpdateCache(channel.Id, stillLiveIDs)
	}

	// Convert map to slice
	var result []*youtube.Video
	for _, video := range allLiveVideos {
		result = append(result, video)
	}

	log.Printf("Total live streams for channel %s: %d (recent: %d, cached: %d)",
		channel.Id, len(result), len(recentLiveVideos), len(result)-len(recentLiveVideos))

	return result, nil
}

// processChannelData processes YouTube API response into a channel snapshot
func processChannelData(ctx context.Context, ytSvc *youtube.Service, channel *youtube.Channel, channelID string, disableLive bool) (ytChannelSnapshot, error) {
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

	// Fetch live streams using the caching + efficient method
	// Pass the channel object which already has contentDetails and disableLive flag
	liveVideos, err := fetchChannelLiveStreams(ctx, ytSvc, channel, disableLive)
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
