FROM golang:1.23-alpine AS builder
WORKDIR /build
COPY go.mod go.sum* ./
RUN go mod tidy
COPY . .
RUN CGO_ENABLED=0 GOFLAGS=-mod=mod go build -ldflags="-s -w" -o /fde-rag ./cmd/fde-rag

FROM scratch
COPY --from=builder /fde-rag /fde-rag
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
ENTRYPOINT ["/fde-rag"]
