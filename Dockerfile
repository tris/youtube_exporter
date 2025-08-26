FROM golang:1.24-alpine AS builder
LABEL org.opencontainers.image.authors="Tristan Horn <tristan+docker@ethereal.net>"
WORKDIR /app
RUN apk add --no-cache upx
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -a -installsuffix cgo -o youtube_exporter .
RUN upx --lzma youtube_exporter

FROM scratch
COPY --from=builder /app/youtube_exporter /youtube_exporter
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
ENTRYPOINT ["/youtube_exporter"]
