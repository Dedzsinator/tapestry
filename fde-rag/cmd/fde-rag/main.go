package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	neo4jdriver "github.com/neo4j/neo4j-go-driver/v5/neo4j"
	pb "github.com/qdrant/go-client/qdrant"
	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ── config ────────────────────────────────────────────────────────────────────

var (
	embedURL   = getenv("EMBED_URL", "http://localhost:7910")
	qdrantURL  = getenv("QDRANT_URL", "http://localhost:6333")
	neo4jURI   = getenv("NEO4J_URI", "bolt://localhost:7687")
	neo4jUser  = getenv("NEO4J_USER", "neo4j")
	neo4jPass  = getenv("NEO4J_PASSWORD", "")
	qwenURL    = getenv("QWEN_URL", "http://192.168.100.10:8001/v1")
	qwenModel  = getenv("QWEN_MODEL", "Qwen/Qwen3.8-27B-FP8")
	apiKey     = getenv("SPARK_VLLM_API_KEY", "local-dummy")
	port       = getenv("PORT", "7900")
	collection = "obsidian_hybrid"
)

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

// ── Qdrant gRPC client ────────────────────────────────────────────────────────

var qdrantClient pb.PointsClient

func initQdrant() {
	addr := strings.TrimPrefix(strings.TrimPrefix(qdrantURL, "https://"), "http://")
	if !strings.Contains(addr, ":") {
		addr += ":6334"
	} else {
		// replace HTTP port with gRPC port
		parts := strings.SplitN(addr, ":", 2)
		addr = parts[0] + ":6334"
	}
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("qdrant gRPC dial: %v", err)
	}
	qdrantClient = pb.NewPointsClient(conn)
}

// ── embedding helpers (call fde-ingest) ───────────────────────────────────────

func httpPost(url string, body any, dst any) error {
	b, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(b)) //nolint:gosec
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return json.Unmarshal(raw, dst)
}

func embedDense(text string) ([]float32, error) {
	var r struct{ Vector []float32 }
	err := httpPost(embedURL+"/embed", map[string]string{"text": text}, &r)
	return r.Vector, err
}

func embedSparse(text string) ([]uint32, []float32, error) {
	var r struct {
		Indices []uint32
		Values  []float32
	}
	err := httpPost(embedURL+"/embed-sparse", map[string]string{"text": text}, &r)
	return r.Indices, r.Values, err
}

func rerank(query string, passages []string) ([]float64, error) {
	var r struct{ Scores []float64 }
	err := httpPost(embedURL+"/rerank", map[string]any{"query": query, "passages": passages}, &r)
	return r.Scores, err
}

// ── Neo4j ─────────────────────────────────────────────────────────────────────

var neo4jDriver neo4jdriver.DriverWithContext

func initNeo4j() {
	d, err := neo4jdriver.NewDriverWithContext(neo4jURI, neo4jdriver.BasicAuth(neo4jUser, neo4jPass, ""))
	if err != nil {
		log.Fatalf("neo4j driver: %v", err)
	}
	neo4jDriver = d
}

func graphNeighbors(ctx context.Context, paths []string) []string {
	session := neo4jDriver.NewSession(ctx, neo4jdriver.SessionConfig{DatabaseName: "fde-rag"})
	defer session.Close(ctx)
	result, err := session.Run(ctx,
		"MATCH (n:Note)-[:LINKS_TO]->(m:Note) WHERE n.path IN $paths RETURN DISTINCT m.path AS p",
		map[string]any{"paths": paths},
	)
	if err != nil {
		return nil
	}
	var out []string
	for result.Next(ctx) {
		if v, ok := result.Record().Get("p"); ok {
			out = append(out, v.(string))
		}
	}
	return out
}

// ── OpenAI (Qwen) ─────────────────────────────────────────────────────────────

var llmClient *openai.Client

func initLLM() {
	cfg := openai.DefaultConfig(apiKey)
	cfg.BaseURL = qwenURL
	llmClient = openai.NewClientWithConfig(cfg)
}

// ── hybrid search ─────────────────────────────────────────────────────────────

type Hit struct {
	Path  string
	Text  string
	Score float64
}

func hybridSearch(ctx context.Context, query string, limit int, graph bool) ([]Hit, error) {
	dv, err := embedDense(query)
	if err != nil {
		return nil, fmt.Errorf("dense embed: %w", err)
	}
	idx, vals, err := embedSparse(query)
	if err != nil {
		return nil, fmt.Errorf("sparse embed: %w", err)
	}

	resp, err := qdrantClient.Query(ctx, &pb.QueryPoints{
		CollectionName: collection,
		Prefetch: []*pb.PrefetchQuery{
			{
				Query: &pb.Query{Variant: &pb.Query_Nearest{Nearest: &pb.VectorInput{
					Variant: &pb.VectorInput_Dense{Dense: &pb.DenseVector{Data: dv}},
				}}},
				Using: strPtr("dense"),
				Limit: uint64Ptr(20),
			},
			{
				Query: &pb.Query{Variant: &pb.Query_Nearest{Nearest: &pb.VectorInput{
					Variant: &pb.VectorInput_Sparse{Sparse: &pb.SparseVector{
						Indices: idx,
						Values:  vals,
					}},
				}}},
				Using: strPtr("sparse"),
				Limit: uint64Ptr(20),
			},
		},
		Query: &pb.Query{Variant: &pb.Query_Fusion{Fusion: pb.Fusion_RRF}},
		Limit: uint64Ptr(20),
		WithPayload: &pb.WithPayloadSelector{
			SelectorOptions: &pb.WithPayloadSelector_Enable{Enable: true},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant query: %w", err)
	}

	hits := make([]Hit, 0, len(resp.Result))
	texts := make([]string, 0, len(resp.Result))
	for _, pt := range resp.Result {
		p := payloadStr(pt.Payload, "path")
		t := payloadStr(pt.Payload, "text")
		hits = append(hits, Hit{Path: p, Text: t})
		texts = append(texts, t)
	}

	if graph && len(hits) > 0 {
		paths := uniquePaths(hits)
		neighbors := graphNeighbors(ctx, paths)
		_ = neighbors // neighbor paths available for future expansion
	}

	scores, err := rerank(query, texts)
	if err == nil && len(scores) == len(hits) {
		for i := range hits {
			hits[i].Score = scores[i]
		}
		sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	}

	if len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

// ── LLM answer ────────────────────────────────────────────────────────────────

func askVault(ctx context.Context, question string) (string, []string, error) {
	hits, err := hybridSearch(ctx, question, 8, true)
	if err != nil {
		return "", nil, err
	}
	var sb strings.Builder
	for _, h := range hits {
		fmt.Fprintf(&sb, "[%s]\n%s\n\n", h.Path, h.Text)
	}
	resp, err := llmClient.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: qwenModel,
		Messages: []openai.ChatCompletionMessage{
			{Role: "system", Content: "Answer the question using only the provided vault context. Be concise."},
			{Role: "user", Content: "Context:\n" + sb.String() + "\nQuestion: " + question},
		},
		MaxTokens: 2048,
	})
	if err != nil {
		return "", nil, fmt.Errorf("llm: %w", err)
	}
	sources := make([]string, len(hits))
	for i, h := range hits {
		sources[i] = h.Path
	}
	return resp.Choices[0].Message.Content, sources, nil
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, map[string]bool{"ok": true})
}

func handleSearch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
		Graph bool   `json:"graph"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if req.Limit == 0 {
		req.Limit = 8
	}
	hits, err := hybridSearch(r.Context(), req.Query, req.Limit, req.Graph)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeJSON(w, map[string]any{"results": hits})
}

func handleAsk(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query string `json:"query"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	answer, sources, err := askVault(r.Context(), req.Query)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeJSON(w, map[string]any{"answer": answer, "sources": sources})
}

// ── MCP tools ─────────────────────────────────────────────────────────────────

func mcpSearchVault(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args, _ := req.Params.Arguments.(map[string]any)
	query, _ := args["query"].(string)
	limitF, _ := args["limit"].(float64)
	limit := int(limitF)
	if limit == 0 {
		limit = 8
	}
	hits, err := hybridSearch(ctx, query, limit, true)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	var sb strings.Builder
	for _, h := range hits {
		fmt.Fprintf(&sb, "[%s | %.3f]\n%s\n\n", h.Path, h.Score, h.Text)
	}
	return mcp.NewToolResultText(sb.String()), nil
}

func mcpAskVault(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	args, _ := req.Params.Arguments.(map[string]any)
	question, _ := args["question"].(string)
	answer, sources, err := askVault(ctx, question)
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	out := answer + "\n\nSources: " + strings.Join(sources, ", ")
	return mcp.NewToolResultText(out), nil
}

func mcpVaultStatus(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	resp, err := qdrantClient.Count(ctx, &pb.CountPoints{
		CollectionName: collection,
		Exact:          boolPtr(false),
	})
	if err != nil {
		return mcp.NewToolResultError(err.Error()), nil
	}
	return mcp.NewToolResultText(fmt.Sprintf("Collection '%s': %d chunks indexed", collection, resp.Result.Count)), nil
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	initQdrant()
	initNeo4j()
	initLLM()

	// Wait for fde-ingest embed service to be ready
	for i := 0; i < 30; i++ {
		resp, err := http.Get(embedURL + "/health") //nolint:gosec
		if err == nil && resp.StatusCode == 200 {
			resp.Body.Close()
			log.Println("[fde-rag] embed service ready")
			break
		}
		log.Printf("[fde-rag] waiting for embed service (%d/30)...", i+1)
		time.Sleep(5 * time.Second)
	}

	// MCP server
	mcpSrv := server.NewMCPServer("fde-rag", "1.0.0",
		server.WithToolCapabilities(false),
	)
	mcpSrv.AddTool(mcp.NewTool("search_vault",
		mcp.WithDescription("Hybrid semantic+sparse search over the Obsidian vault"),
		mcp.WithString("query", mcp.Required(), mcp.Description("Search query")),
		mcp.WithNumber("limit", mcp.Description("Max results (default 8)")),
	), mcpSearchVault)
	mcpSrv.AddTool(mcp.NewTool("ask_vault",
		mcp.WithDescription("Ask a question answered by the Obsidian vault via RAG"),
		mcp.WithString("question", mcp.Required(), mcp.Description("Question to answer")),
	), mcpAskVault)
	mcpSrv.AddTool(mcp.NewTool("vault_status",
		mcp.WithDescription("Return indexed chunk count for the vault collection"),
	), mcpVaultStatus)

	// HTTP mux
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handleHealth)
	mux.HandleFunc("POST /search", handleSearch)
	mux.HandleFunc("POST /ask", handleAsk)

	// Mount MCP at /mcp (streamable-http)
	httpSrv := server.NewStreamableHTTPServer(mcpSrv)
	mux.Handle("/mcp", httpSrv)
	mux.Handle("/mcp/", httpSrv)

	addr := "0.0.0.0:" + port
	log.Printf("[fde-rag] listening on %s (MCP at /mcp)", addr)
	if err := http.ListenAndServe(addr, mux); err != nil { //nolint:gosec
		log.Fatal(err)
	}
}

// ── small helpers ─────────────────────────────────────────────────────────────

func payloadStr(payload map[string]*pb.Value, key string) string {
	if v, ok := payload[key]; ok {
		if sv := v.GetStringValue(); sv != "" {
			return sv
		}
	}
	return ""
}

func uniquePaths(hits []Hit) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, h := range hits {
		if !seen[h.Path] {
			seen[h.Path] = true
			out = append(out, h.Path)
		}
	}
	return out
}

func strPtr(s string) *string   { return &s }
func boolPtr(b bool) *bool      { return &b }
func uint64Ptr(n uint64) *uint64 { return &n }
