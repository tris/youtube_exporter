package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	defaultPort = 9473
)

func main() {
	apiKey := os.Getenv("YOUTUBE_API_KEY")
	if apiKey == "" {
		log.Fatal("You must provide an API key via YOUTUBE_API_KEY")
	}

	ctx := context.Background()
	ytSvc, err := newYouTubeService(ctx, apiKey)
	if err != nil {
		log.Fatalf("main: Failed to initialize YouTube service: %v", err)
	}

	// Register HTTP handlers
	http.HandleFunc("/metrics", promhttp.HandlerFor(prometheus.DefaultGatherer, promhttp.HandlerOpts{}).ServeHTTP)
	http.HandleFunc("/scrape", metricsHandler(ytSvc))

	port := os.Getenv("PORT")
	if port == "" {
		port = strconv.Itoa(defaultPort)
	}

	log.Printf("Listening on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
