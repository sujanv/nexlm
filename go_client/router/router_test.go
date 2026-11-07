package router

import (
	"errors"
	"sync"
	"testing"
	"time"
)

func TestTokenBucketRateLimiter(t *testing.T) {
	// 10 tokens/sec, burst capacity of 2
	limiter := NewTokenBucket(10.0, 2.0)

	// First two requests consume burst capacity
	if !limiter.Allow() {
		t.Fatalf("first request should be allowed")
	}
	if !limiter.Allow() {
		t.Fatalf("second burst request should be allowed")
	}

	// Immediate third request should be rejected (bucket empty)
	if limiter.Allow() {
		t.Fatalf("third immediate request should exceed rate limit")
	}

	// Sleep 120ms to replenish at least 1 token (10 tokens/sec = 1 token per 100ms)
	time.Sleep(120 * time.Millisecond)
	if !limiter.Allow() {
		t.Fatalf("request after refill interval should be allowed")
	}
}

func TestRouterRoundRobin(t *testing.T) {
	b1 := &Backend{URL: "http://localhost:8080", Healthy: true}
	b2 := &Backend{URL: "http://localhost:8081", Healthy: true}

	r := NewRouter([]*Backend{b1, b2}, 100.0, 100.0, "round_robin")

	pick1, _ := r.PickBackend()
	pick2, _ := r.PickBackend()
	pick3, _ := r.PickBackend()

	if pick1.URL != b1.URL || pick2.URL != b2.URL || pick3.URL != b1.URL {
		t.Fatalf("unexpected round robin sequence: %s, %s, %s", pick1.URL, pick2.URL, pick3.URL)
	}
}

func TestRouterLeastConnection(t *testing.T) {
	b1 := &Backend{URL: "http://localhost:8080", Healthy: true, InFlight: 5}
	b2 := &Backend{URL: "http://localhost:8081", Healthy: true, InFlight: 1}

	r := NewRouter([]*Backend{b1, b2}, 100.0, 100.0, "least_conn")

	pick, err := r.PickBackend()
	if err != nil {
		t.Fatalf("failed to pick backend: %v", err)
	}
	if pick.URL != b2.URL {
		t.Fatalf("expected backend with fewest connections (b2), got %s", pick.URL)
	}
}

func TestRouterUnhealthySkip(t *testing.T) {
	b1 := &Backend{URL: "http://localhost:8080", Healthy: false}
	b2 := &Backend{URL: "http://localhost:8081", Healthy: true}

	r := NewRouter([]*Backend{b1, b2}, 100.0, 100.0, "round_robin")

	pick, err := r.PickBackend()
	if err != nil {
		t.Fatalf("failed to pick backend: %v", err)
	}
	if pick.URL != b2.URL {
		t.Fatalf("expected healthy backend b2, got %s", pick.URL)
	}

	// Mark all unhealthy
	b2.SetHealthy(false)
	_, err = r.PickBackend()
	if !errors.Is(err, ErrNoHealthyBackend) {
		t.Fatalf("expected ErrNoHealthyBackend when all nodes down, got %v", err)
	}
}

func TestRouterExecuteRequestConcurrent(t *testing.T) {
	b1 := &Backend{URL: "http://localhost:8080", Healthy: true}
	b2 := &Backend{URL: "http://localhost:8081", Healthy: true}

	r := NewRouter([]*Backend{b1, b2}, 1000.0, 100.0, "round_robin")

	var wg sync.WaitGroup
	numRequests := 50

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := r.ExecuteRequest(func(b *Backend) (string, error) {
				time.Sleep(2 * time.Millisecond)
				return "ok", nil
			})
			if err != nil {
				t.Errorf("request failed: %v", err)
			}
		}()
	}

	wg.Wait()

	total := b1.TotalHandled + b2.TotalHandled
	if total != int64(numRequests) {
		t.Fatalf("expected %d total handled requests, got %d", numRequests, total)
	}
}
