package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

type CompletionRequest struct {
	Prompt      string  `json:"prompt"`
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k"`
	MaxTokens   int     `json:"max_tokens"`
	Stream      bool    `json:"stream"`
}

type StreamEvent struct {
	Token string `json:"token"`
}

var httpClient = &http.Client{
	Transport: &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     90 * time.Second,
	},
	Timeout: 120 * time.Second,
}

func streamGeneration(serverURL, prompt string, temp float64, maxTokens int) error {
	reqBody := CompletionRequest{
		Prompt:      prompt,
		Temperature: temp,
		TopK:        40,
		MaxTokens:   maxTokens,
		Stream:      true,
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return err
	}

	endpoint := strings.TrimRight(serverURL, "/") + "/v1/chat/completions"
	req, err := http.NewRequest("POST", endpoint, bytes.NewBuffer(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	t0 := time.Now()
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("connection failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("HTTP error %d: %s", resp.StatusCode, string(b))
	}

	fmt.Printf("\033[1m\033[96mPrompt: %s\033[0m\n\n\033[92m", prompt)

	scanner := bufio.NewScanner(resp.Body)
	var firstTokenTime time.Duration
	tokenCount := 0

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var event StreamEvent
			if err := json.Unmarshal([]byte(data), &event); err == nil {
				if tokenCount == 0 {
					firstTokenTime = time.Since(t0)
				}
				tokenCount++
				fmt.Print(event.Token)
			}
		}
	}

	elapsed := time.Since(t0)
	fmt.Printf("\033[0m\n\n\033[2m--- Stats: %d tokens | TTFT: %v | Total: %v (%.1f tok/s) ---\033[0m\n",
		tokenCount, firstTokenTime.Round(time.Millisecond), elapsed.Round(time.Millisecond),
		float64(tokenCount)/elapsed.Seconds())

	return scanner.Err()
}

func runBenchmark(serverURL string, concurrency, numRequests, tokensPerReq int) {
	fmt.Printf("⚡ Running NexLM Go Benchmark: %d requests, Concurrency %d\n", numRequests, concurrency)

	var wg sync.WaitGroup
	ch := make(chan string, numRequests)
	for i := 0; i < numRequests; i++ {
		ch <- fmt.Sprintf("Once upon a time, request %d", i+1)
	}
	close(ch)

	t0 := time.Now()
	var totalTokens int64
	var mu sync.Mutex

	for c := 0; c < concurrency; c++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for prompt := range ch {
				reqBody := CompletionRequest{
					Prompt:      prompt,
					Temperature: 0.7,
					TopK:        40,
					MaxTokens:   tokensPerReq,
					Stream:      false,
				}
				b, _ := json.Marshal(reqBody)
				req, _ := http.NewRequest("POST", serverURL+"/v1/chat/completions", bytes.NewBuffer(b))
				req.Header.Set("Content-Type", "application/json")

				resp, err := httpClient.Do(req)
				if err == nil && resp.StatusCode == 200 {
					var resMap map[string]interface{}
					json.NewDecoder(resp.Body).Decode(&resMap)
					resp.Body.Close()
					mu.Lock()
					totalTokens += int64(tokensPerReq)
					mu.Unlock()
				}
			}
		}()
	}

	wg.Wait()
	elapsed := time.Since(t0)
	fmt.Printf("\n✓ Completed in %v | Total Tokens: %d | Throughput: %.1f tokens/sec\n",
		elapsed.Round(time.Millisecond), totalTokens, float64(totalTokens)/elapsed.Seconds())
}

func main() {
	serverURL := flag.String("server", "http://127.0.0.1:8080", "NexLM server base URL")
	prompt := flag.String("prompt", "Once upon a time, Lily found a golden key", "Generation prompt")
	temp := flag.Float64("temp", 0.7, "Sampling temperature")
	maxTokens := flag.Int("tokens", 60, "Max new tokens")
	bench := flag.Bool("bench", false, "Run concurrency benchmark")
	concurrency := flag.Int("c", 4, "Benchmark concurrency")
	requests := flag.Int("n", 20, "Benchmark total requests")
	flag.Parse()

	if *bench {
		runBenchmark(*serverURL, *concurrency, *requests, *maxTokens)
	} else {
		if err := streamGeneration(*serverURL, *prompt, *temp, *maxTokens); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}
}
