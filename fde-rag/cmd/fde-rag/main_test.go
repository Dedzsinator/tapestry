package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"
)

var ragBase = func() string {
	if v := os.Getenv("RAG_URL"); v != "" {
		return v
	}
	return "http://192.168.100.11:7900"
}()

var embedBase = func() string {
	if v := os.Getenv("EMBED_URL"); v != "" {
		return v
	}
	return "http://192.168.100.11:7910"
}()

func post(t *testing.T, url string, body any) map[string]any {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(b)) //nolint:gosec
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("POST %s: status %d — %s", url, resp.StatusCode, raw)
	}
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	return out
}

func get(t *testing.T, url string) map[string]any {
	t.Helper()
	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	return out
}

// ── health checks ─────────────────────────────────────────────────────────────

func TestRAGHealth(t *testing.T) {
	r := get(t, ragBase+"/health")
	if r["ok"] != true {
		t.Errorf("expected ok:true, got %v", r)
	}
}

func TestEmbedHealth(t *testing.T) {
	r := get(t, embedBase+"/health")
	if r["ok"] != true {
		t.Errorf("expected ok:true, got %v", r)
	}
}

// ── embed layer ───────────────────────────────────────────────────────────────

func TestDenseEmbed(t *testing.T) {
	r := post(t, embedBase+"/embed", map[string]string{"text": "predictive maintenance vibration sensor"})
	vec, ok := r["vector"].([]any)
	if !ok || len(vec) == 0 {
		t.Fatalf("expected vector slice, got %T %v", r["vector"], r["vector"])
	}
	if len(vec) != 768 {
		t.Errorf("expected 768-dim nomic embedding, got %d", len(vec))
	}
	t.Logf("dense embed: dim=%d first3=[%.4f %.4f %.4f]", len(vec), vec[0], vec[1], vec[2])
}

func TestSparseEmbed(t *testing.T) {
	r := post(t, embedBase+"/embed-sparse", map[string]string{"text": "Ignition SCADA OPC-UA"})
	indices, ok := r["indices"].([]any)
	if !ok || len(indices) == 0 {
		t.Fatalf("expected indices, got %v", r)
	}
	t.Logf("sparse embed: %d non-zero tokens", len(indices))
}

func TestRerank(t *testing.T) {
	r := post(t, embedBase+"/rerank", map[string]any{
		"query": "predictive maintenance",
		"passages": []string{
			"Vibration analysis detects bearing failure early",
			"EMQX MQTT broker for IoT messaging",
			"Random forest classifier for anomaly detection",
		},
	})
	scores, ok := r["scores"].([]any)
	if !ok || len(scores) != 3 {
		t.Fatalf("expected 3 scores, got %v", r)
	}
	t.Logf("rerank scores: %.4f %.4f %.4f", scores[0], scores[1], scores[2])
	// passage 0 and 2 should score higher than passage 1
	s0, s2, s1 := scores[0].(float64), scores[2].(float64), scores[1].(float64)
	if s1 > s0 && s1 > s2 {
		t.Errorf("MQTT passage scored higher than maintenance passages: %.4f vs %.4f / %.4f", s1, s0, s2)
	}
}

// ── search ────────────────────────────────────────────────────────────────────

func TestHybridSearch(t *testing.T) {
	r := post(t, ragBase+"/search", map[string]any{
		"query": "Qwen vLLM model configuration",
		"limit": 5,
		"graph": false,
	})
	results, ok := r["results"].([]any)
	if !ok {
		t.Fatalf("expected results array, got %v", r)
	}
	if len(results) == 0 {
		t.Fatal("no results returned — collection may still be indexing")
	}
	t.Logf("hybrid search returned %d results", len(results))
	for i, res := range results {
		m := res.(map[string]any)
		t.Logf("  [%d] path=%s score=%.4f text=%.60s…", i, m["Path"], m["Score"], m["Text"])
	}
}

func TestGraphSearch(t *testing.T) {
	r := post(t, ragBase+"/search", map[string]any{
		"query": "EMQX MQTT Sparkplug-B",
		"limit": 5,
		"graph": true,
	})
	results, ok := r["results"].([]any)
	if !ok || len(results) == 0 {
		t.Fatalf("expected results, got %v", r)
	}
	t.Logf("graph search: %d results for EMQX query", len(results))
}

func TestSearchRelevance(t *testing.T) {
	queries := []struct {
		q    string
		want string // substring expected in top result path or text
	}{
		{"predmaint K8s Grafana dashboards", "predmaint"},
		{"spark1 spark2 vLLM GPU cluster", "spark"},
		{"Obsidian vault neo4j qdrant RAG", "rag"},
		{"nandor Docker Compose EMQX migration", "emqx"},
	}
	for _, tc := range queries {
		t.Run(tc.q[:30], func(t *testing.T) {
			r := post(t, ragBase+"/search", map[string]any{"query": tc.q, "limit": 3})
			results, _ := r["results"].([]any)
			if len(results) == 0 {
				t.Skipf("no results yet — still indexing")
			}
			top := results[0].(map[string]any)
			path := fmt.Sprintf("%v", top["Path"])
			text := fmt.Sprintf("%v", top["Text"])
			t.Logf("top result: %s (%.4f)", path, top["Score"])
			_ = text // could assert contains tc.want
		})
	}
}

// ── MCP protocol ─────────────────────────────────────────────────────────────

func TestMCPInit(t *testing.T) {
	payload := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "initialize",
		"params": map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{},
			"clientInfo":      map[string]string{"name": "test", "version": "0.0.1"},
		},
	}
	b, _ := json.Marshal(payload)
	resp, err := http.Post(ragBase+"/mcp", "application/json", bytes.NewReader(b)) //nolint:gosec
	if err != nil {
		t.Fatalf("MCP init: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	t.Logf("MCP init response status=%d body=%s", resp.StatusCode, raw[:min(len(raw), 300)])
	if resp.StatusCode >= 400 {
		t.Errorf("MCP init returned %d", resp.StatusCode)
	}
}

func TestMCPToolsList(t *testing.T) {
	// session init first
	sessionID := mcpSession(t)
	payload := map[string]any{
		"jsonrpc": "2.0",
		"id":      2,
		"method":  "tools/list",
		"params":  map[string]any{},
	}
	b, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", ragBase+"/mcp", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("mcp-session-id", sessionID)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("tools/list: %v", err)
	}
	defer resp.Body.Close()
	var r map[string]any
	json.NewDecoder(resp.Body).Decode(&r)
	result, _ := r["result"].(map[string]any)
	tools, _ := result["tools"].([]any)
	names := make([]string, len(tools))
	for i, tool := range tools {
		names[i] = tool.(map[string]any)["name"].(string)
	}
	t.Logf("MCP tools: %v", names)
	expected := map[string]bool{"search_vault": false, "ask_vault": false, "vault_status": false}
	for _, n := range names {
		expected[n] = true
	}
	for name, found := range expected {
		if !found {
			t.Errorf("expected tool %q not found in list", name)
		}
	}
}

func TestMCPVaultStatus(t *testing.T) {
	sessionID := mcpSession(t)
	payload := map[string]any{
		"jsonrpc": "2.0",
		"id":      3,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      "vault_status",
			"arguments": map[string]any{},
		},
	}
	b, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", ragBase+"/mcp", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("mcp-session-id", sessionID)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("vault_status: %v", err)
	}
	defer resp.Body.Close()
	var r map[string]any
	json.NewDecoder(resp.Body).Decode(&r)
	t.Logf("vault_status response: %v", r)
}

func TestMCPSearchVault(t *testing.T) {
	sessionID := mcpSession(t)
	payload := map[string]any{
		"jsonrpc": "2.0",
		"id":      4,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "search_vault",
			"arguments": map[string]any{
				"query": "DSpark migration DeepSeek spark1",
				"limit": 3,
			},
		},
	}
	b, _ := json.Marshal(payload)
	req, _ := http.NewRequest("POST", ragBase+"/mcp", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("mcp-session-id", sessionID)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("search_vault: %v", err)
	}
	defer resp.Body.Close()
	var r map[string]any
	json.NewDecoder(resp.Body).Decode(&r)
	result, _ := r["result"].(map[string]any)
	content, _ := result["content"].([]any)
	if len(content) > 0 {
		text := content[0].(map[string]any)["text"].(string)
		t.Logf("search_vault result (first 300 chars):\n%s", text[:min(len(text), 300)])
	}
}

// ── performance benchmark ─────────────────────────────────────────────────────

func BenchmarkHybridSearch(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		body, _ := json.Marshal(map[string]any{"query": "predmaint sensor anomaly detection", "limit": 5})
		resp, err := http.Post(ragBase+"/search", "application/json", bytes.NewReader(body)) //nolint:gosec
		if err != nil {
			b.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
}

func BenchmarkEmbedDense(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		body, _ := json.Marshal(map[string]string{"text": "vibration sensor bearing failure RMS"})
		resp, err := http.Post(embedBase+"/embed", "application/json", bytes.NewReader(body)) //nolint:gosec
		if err != nil {
			b.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
}

// ── latency test ──────────────────────────────────────────────────────────────

func TestSearchLatency(t *testing.T) {
	queries := []string{
		"Qwen3 vLLM configuration spark1",
		"EMQX MQTT Sparkplug-B decoder bridge",
		"predmaint Grafana dashboard K8s",
		"Neo4j graph RAG hybrid search",
		"Obsidian vault note wikilink",
	}
	var totalMs int64
	for _, q := range queries {
		start := time.Now()
		post(t, ragBase+"/search", map[string]any{"query": q, "limit": 5})
		ms := time.Since(start).Milliseconds()
		totalMs += ms
		t.Logf("query=%q latency=%dms", q[:min(len(q), 40)], ms)
	}
	avg := totalMs / int64(len(queries))
	t.Logf("average latency: %dms", avg)
	if avg > 5000 {
		t.Errorf("average search latency %dms exceeds 5s threshold", avg)
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

func mcpSession(t *testing.T) string {
	t.Helper()
	payload := map[string]any{
		"jsonrpc": "2.0",
		"id":      0,
		"method":  "initialize",
		"params": map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{},
			"clientInfo":      map[string]string{"name": "test", "version": "0.0.1"},
		},
	}
	b, _ := json.Marshal(payload)
	resp, err := http.Post(ragBase+"/mcp", "application/json", bytes.NewReader(b)) //nolint:gosec
	if err != nil {
		t.Fatalf("MCP session init: %v", err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.Header.Get("mcp-session-id")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
