package router

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

var (
	ErrRateLimited      = errors.New("rate limit exceeded")
	ErrNoHealthyBackend = errors.New("no healthy backends available")
)

// TokenBucket implements a thread-safe token bucket rate limiter.
type TokenBucket struct {
	mu         sync.Mutex
	capacity   float64
	tokens     float64
	refillRate float64 // tokens per second
	lastRefill time.Time
}

// NewTokenBucket creates a new rate limiter with specified QPS and burst capacity.
func NewTokenBucket(ratePerSec float64, capacity float64) *TokenBucket {
	return &TokenBucket{
		capacity:   capacity,
		tokens:     capacity,
		refillRate: ratePerSec,
		lastRefill: time.Now(),
	}
}

// Allow checks if 1 token is available, replenishes tokens based on elapsed time, and consumes it.
func (tb *TokenBucket) Allow() bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tb.lastRefill = now

	// Replenish tokens
	tb.tokens += elapsed * tb.refillRate
	if tb.tokens > tb.capacity {
		tb.tokens = tb.capacity
	}

	if tb.tokens >= 1.0 {
		tb.tokens -= 1.0
		return true
	}
	return false
}

// Backend represents a single NexLM inference server instance.
type Backend struct {
	URL            string
	Healthy        bool
	InFlight       int64
	TotalHandled   int64
	mu             sync.RWMutex
}

// SetHealthy updates the health status of a backend.
func (b *Backend) SetHealthy(healthy bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.Healthy = healthy
}

// IsHealthy returns whether the backend is ready to accept traffic.
func (b *Backend) IsHealthy() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.Healthy
}

// Router routes requests across multiple NexLM backend nodes with load balancing and rate limiting.
type Router struct {
	backends    []*Backend
	limiter     *TokenBucket
	nextIndex   uint64
	strategy    string // "round_robin" or "least_conn"
	mu          sync.RWMutex
}

// NewRouter constructs a cluster router.
func NewRouter(backends []*Backend, qps float64, burst float64, strategy string) *Router {
	return &Router{
		backends: backends,
		limiter:  NewTokenBucket(qps, burst),
		strategy: strategy,
	}
}

// PickBackend selects a healthy backend using the configured strategy.
func (r *Router) PickBackend() (*Backend, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var healthy []*Backend
	for _, b := range r.backends {
		if b.IsHealthy() {
			healthy = append(healthy, b)
		}
	}

	if len(healthy) == 0 {
		return nil, ErrNoHealthyBackend
	}

	if r.strategy == "least_conn" {
		best := healthy[0]
		minConn := atomic.LoadInt64(&best.InFlight)
		for _, b := range healthy[1:] {
			inFlight := atomic.LoadInt64(&b.InFlight)
			if inFlight < minConn {
				minConn = inFlight
				best = b
			}
		}
		return best, nil
	}

	// Default: Round Robin
	idx := atomic.AddUint64(&r.nextIndex, 1) - 1
	return healthy[idx%uint64(len(healthy))], nil
}

// ExecuteRequest wraps a request dispatch with rate-limiting and connection tracking.
func (r *Router) ExecuteRequest(handler func(b *Backend) (string, error)) (string, error) {
	if r.limiter != nil && !r.limiter.Allow() {
		return "", ErrRateLimited
	}

	backend, err := r.PickBackend()
	if err != nil {
		return "", err
	}

	atomic.AddInt64(&backend.InFlight, 1)
	defer atomic.AddInt64(&backend.InFlight, -1)
	defer atomic.AddInt64(&backend.TotalHandled, 1)

	return handler(backend)
}
