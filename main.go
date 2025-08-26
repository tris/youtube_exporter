package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
)

var (
	addrFlag   = flag.String("addr", ":9473", "listen address")
	apiKeyFlag = flag.String("api-key", "", "YouTube Data API key (overrides YOUTUBE_API_KEY)")
)

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
	ytSvc, err := newYouTubeService(ctx, apiKey)
	if err != nil {
		log.Fatalf("youtube.NewService: %v", err)
	}

	// Register HTTP handlers
	http.HandleFunc("/metrics", metricsHandler(ytSvc))

	log.Printf("Listening on %s …", *addrFlag)
	if err := http.ListenAndServe(*addrFlag, nil); err != nil {
		log.Fatal(err)
	}
}
