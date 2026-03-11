package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

type GenerateRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float32 `json:"temperature,omitempty"`
}

type GenerateResponse struct {
	Response string `json:"response"`
}

func main() {
	base := os.Getenv("LOCI_BASE_URL")
	if base == "" {
		base = "http://127.0.0.1:8080"
	}

	health, err := httpGet(base + "/v1/health")
	if err != nil {
		panic(err)
	}
	fmt.Println("health:", health)

	info, err := httpGet(base + "/v1/info")
	if err != nil {
		panic(err)
	}
	fmt.Println("info:", info)

	resp, err := generate(base, GenerateRequest{
		Prompt:      "Hello from Go template",
		MaxTokens:   8,
		Temperature: 0.7,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println("response:", resp.Response)
}

func httpGet(url string) (string, error) {
	r, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return "", err
	}
	if r.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GET %s failed: %s %s", url, r.Status, string(body))
	}
	return string(body), nil
}

func generate(base string, req GenerateRequest) (*GenerateResponse, error) {
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.Post(base+"/v1/generate", "application/json", bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()

	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	if r.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("POST /v1/generate failed: %s %s", r.Status, string(body))
	}

	var out GenerateResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
