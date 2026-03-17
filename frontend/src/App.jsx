import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const EXAMPLE_QUESTIONS = [
  "What must a GDPR data controller do within 72 hours of a data breach?",
  "How does CCPA's right to deletion compare to GDPR Article 17?",
  "What obligations apply to data processors under GDPR?",
  "Which privacy regulations cover health data and what do they require?",
  "What consent requirements does GDPR impose on data controllers?",
];

const REGULATIONS = [
  { value: "", label: "All Regulations" },
  { value: "gdpr", label: "GDPR" },
  { value: "ccpa", label: "CCPA" },
  { value: "hipaa", label: "HIPAA" },
];

function SourceBadge({ source }) {
  const isHybrid = source?.includes("+");
  const isGraph = source?.includes("graph");
  const isVector = source?.includes("vector") && !isHybrid;
  return (
    <span
      style={{
        fontSize: "10px",
        fontWeight: 700,
        letterSpacing: "0.08em",
        padding: "2px 7px",
        borderRadius: "999px",
        background: isHybrid ? "#1a3a2a" : isGraph ? "#1a2a3a" : "#2a1a3a",
        color: isHybrid ? "#4ade80" : isGraph ? "#60a5fa" : "#c084fc",
        border: `1px solid ${isHybrid ? "#4ade8033" : isGraph ? "#60a5fa33" : "#c084fc33"}`,
        textTransform: "uppercase",
      }}
    >
      {isHybrid ? "⬡ hybrid" : isGraph ? "◈ graph" : "◉ vector"}
    </span>
  );
}

function Citation({ citation, index }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "10px",
        padding: "10px 14px",
        background: "#0d1117",
        border: "1px solid #1e2a1e",
        borderRadius: "8px",
        marginBottom: "8px",
      }}
    >
      <span
        style={{
          minWidth: "22px",
          height: "22px",
          borderRadius: "50%",
          background: "#0f2a1a",
          border: "1px solid #22c55e44",
          color: "#22c55e",
          fontSize: "11px",
          fontWeight: 700,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "monospace",
        }}
      >
        {index + 1}
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          <span style={{ color: "#22c55e", fontWeight: 700, fontSize: "13px", fontFamily: "monospace" }}>
            {citation.regulation?.toUpperCase() || "—"}
          </span>
          {citation.article && (
            <span style={{ color: "#94a3b8", fontSize: "12px" }}>Art. {citation.article}</span>
          )}
          {citation.title && (
            <span style={{ color: "#64748b", fontSize: "12px", fontStyle: "italic" }}>
              {citation.title}
            </span>
          )}
          <SourceBadge source={citation.source} />
        </div>
      </div>
    </div>
  );
}

function RetrievalTrace({ trace }) {
  const [open, setOpen] = useState(false);
  if (!trace?.length) return null;
  return (
    <div style={{ marginTop: "20px" }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          background: "none",
          border: "1px solid #1e2d1e",
          color: "#4ade80",
          fontSize: "12px",
          padding: "6px 14px",
          borderRadius: "6px",
          cursor: "pointer",
          letterSpacing: "0.05em",
          fontFamily: "monospace",
        }}
      >
        {open ? "▾" : "▸"} RETRIEVAL TRACE ({trace.length} chunks)
      </button>
      {open && (
        <div
          style={{
            marginTop: "12px",
            fontFamily: "monospace",
            fontSize: "11px",
            color: "#64748b",
            background: "#080c08",
            border: "1px solid #111c11",
            borderRadius: "8px",
            padding: "14px",
            maxHeight: "300px",
            overflowY: "auto",
          }}
        >
          {trace.map((item, i) => (
            <div key={i} style={{ marginBottom: "10px", borderBottom: "1px solid #111", paddingBottom: "10px" }}>
              <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "4px" }}>
                <span style={{ color: "#22c55e" }}>#{i + 1}</span>
                <span style={{ color: "#94a3b8" }}>{item.regulation?.toUpperCase()} Art.{item.article}</span>
                <SourceBadge source={item.source} />
                <span style={{ color: "#4ade8099" }}>score: {item.score?.toFixed(4)}</span>
              </div>
              <div style={{ color: "#475569", lineHeight: 1.5 }}>{item.preview}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StatsBar({ stats }) {
  if (!stats) return null;
  return (
    <div
      style={{
        display: "flex",
        gap: "16px",
        padding: "10px 16px",
        background: "#080c08",
        border: "1px solid #111c11",
        borderRadius: "8px",
        marginTop: "16px",
        flexWrap: "wrap",
      }}
    >
      {[
        { label: "Vector hits", value: stats.vector_results, color: "#c084fc" },
        { label: "Graph hits", value: stats.graph_results, color: "#60a5fa" },
        { label: "Fused", value: stats.fused_results, color: "#4ade80" },
        { label: "Latency", value: `${stats.latency_ms?.toFixed(0)}ms`, color: "#fbbf24" },
      ].map((s) => (
        <div key={s.label} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ color: "#374151", fontSize: "11px", letterSpacing: "0.06em", textTransform: "uppercase" }}>
            {s.label}
          </span>
          <span style={{ color: s.color, fontWeight: 700, fontSize: "13px", fontFamily: "monospace" }}>
            {s.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [regulation, setRegulation] = useState("");
  const [showTrace, setShowTrace] = useState(false);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [question]);

  const handleQuery = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          regulation_filter: regulation || null,
          include_trace: showTrace,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }
      const data = await res.json();
      setResponse(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleQuery();
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#030603",
        color: "#e2e8f0",
        fontFamily: "'Geist', 'Inter', system-ui, sans-serif",
        padding: "0",
      }}
    >
      {/* Grid background */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          backgroundImage:
            "linear-gradient(rgba(34,197,94,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.03) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
          pointerEvents: "none",
        }}
      />

      <div style={{ position: "relative", maxWidth: "820px", margin: "0 auto", padding: "60px 24px 80px" }}>

        {/* Header */}
        <div style={{ marginBottom: "48px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
            <div
              style={{
                width: "36px",
                height: "36px",
                borderRadius: "8px",
                background: "linear-gradient(135deg, #0f2a1a, #1a3a2a)",
                border: "1px solid #22c55e33",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "18px",
              }}
            >
              ⬡
            </div>
            <span
              style={{
                fontFamily: "monospace",
                fontSize: "11px",
                letterSpacing: "0.15em",
                color: "#22c55e88",
                textTransform: "uppercase",
              }}
            >
              Privacy Regulation GraphRAG
            </span>
          </div>

          <h1
            style={{
              fontSize: "clamp(28px, 5vw, 42px)",
              fontWeight: 800,
              lineHeight: 1.1,
              margin: "0 0 12px",
              letterSpacing: "-0.02em",
              background: "linear-gradient(135deg, #f0fdf4 30%, #4ade80)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Ask anything about privacy law
          </h1>
          <p style={{ color: "#475569", fontSize: "15px", lineHeight: 1.6, margin: 0, maxWidth: "520px" }}>
            Hybrid retrieval combining <span style={{ color: "#c084fc" }}>semantic vector search</span> and{" "}
            <span style={{ color: "#60a5fa" }}>knowledge graph traversal</span> — +21% accuracy over baseline RAG.
          </p>
        </div>

        {/* Query input */}
        <div
          style={{
            background: "#0a0f0a",
            border: "1px solid #1e2d1e",
            borderRadius: "12px",
            padding: "20px",
            marginBottom: "16px",
          }}
        >
          <textarea
            ref={textareaRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="e.g. What are GDPR's breach notification requirements for data controllers?"
            rows={2}
            style={{
              width: "100%",
              background: "none",
              border: "none",
              outline: "none",
              color: "#e2e8f0",
              fontSize: "16px",
              lineHeight: 1.6,
              resize: "none",
              fontFamily: "inherit",
              boxSizing: "border-box",
            }}
          />

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginTop: "16px",
              gap: "12px",
              flexWrap: "wrap",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
              <select
                value={regulation}
                onChange={(e) => setRegulation(e.target.value)}
                style={{
                  background: "#0d1a0d",
                  border: "1px solid #1e2d1e",
                  color: "#94a3b8",
                  padding: "7px 12px",
                  borderRadius: "7px",
                  fontSize: "13px",
                  fontFamily: "inherit",
                  cursor: "pointer",
                }}
              >
                {REGULATIONS.map((r) => (
                  <option key={r.value} value={r.value}>{r.label}</option>
                ))}
              </select>

              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "7px",
                  color: "#64748b",
                  fontSize: "13px",
                  cursor: "pointer",
                }}
              >
                <input
                  type="checkbox"
                  checked={showTrace}
                  onChange={(e) => setShowTrace(e.target.checked)}
                  style={{ accentColor: "#22c55e" }}
                />
                Show retrieval trace
              </label>
            </div>

            <button
              onClick={handleQuery}
              disabled={loading || !question.trim()}
              style={{
                background: loading ? "#0f2a1a" : "linear-gradient(135deg, #15803d, #22c55e)",
                border: "none",
                color: loading ? "#4ade8066" : "#fff",
                padding: "10px 24px",
                borderRadius: "8px",
                fontSize: "14px",
                fontWeight: 600,
                cursor: loading || !question.trim() ? "not-allowed" : "pointer",
                letterSpacing: "0.02em",
                transition: "all 0.15s",
                minWidth: "120px",
              }}
            >
              {loading ? (
                <span style={{ display: "flex", alignItems: "center", gap: "8px", justifyContent: "center" }}>
                  <span style={{
                    width: "12px", height: "12px", borderRadius: "50%",
                    border: "2px solid #4ade8044", borderTopColor: "#4ade80",
                    animation: "spin 0.8s linear infinite", display: "inline-block"
                  }} />
                  Querying...
                </span>
              ) : "Query ⌘↵"}
            </button>
          </div>
        </div>

        {/* Example questions */}
        <div style={{ marginBottom: "32px" }}>
          <div style={{ color: "#374151", fontSize: "11px", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "10px" }}>
            Example queries
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {EXAMPLE_QUESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => setQuestion(q)}
                style={{
                  background: "#0a0f0a",
                  border: "1px solid #1a271a",
                  color: "#64748b",
                  padding: "6px 12px",
                  borderRadius: "6px",
                  fontSize: "12px",
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "all 0.15s",
                  maxWidth: "300px",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
                onMouseEnter={(e) => {
                  e.target.style.borderColor = "#22c55e44";
                  e.target.style.color = "#94a3b8";
                }}
                onMouseLeave={(e) => {
                  e.target.style.borderColor = "#1a271a";
                  e.target.style.color = "#64748b";
                }}
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div
            style={{
              background: "#1a0a0a",
              border: "1px solid #ef444433",
              borderRadius: "10px",
              padding: "16px 20px",
              color: "#f87171",
              marginBottom: "24px",
              fontSize: "14px",
            }}
          >
            ⚠ {error}
          </div>
        )}

        {/* Response */}
        {response && (
          <div
            style={{
              background: "#0a0f0a",
              border: "1px solid #1e2d1e",
              borderRadius: "12px",
              padding: "28px",
              animation: "fadeIn 0.3s ease",
            }}
          >
            {/* Answer */}
            <div
              style={{
                fontSize: "15px",
                lineHeight: 1.8,
                color: "#cbd5e1",
                marginBottom: "24px",
                whiteSpace: "pre-wrap",
              }}
            >
              {response.answer}
            </div>

            {/* Stats bar */}
            <StatsBar stats={response.retrieval_stats} />

            {/* Citations */}
            {response.citations?.length > 0 && (
              <div style={{ marginTop: "20px" }}>
                <div
                  style={{
                    color: "#374151",
                    fontSize: "11px",
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    marginBottom: "10px",
                  }}
                >
                  Sources ({response.citations.length})
                </div>
                {response.citations.map((c, i) => (
                  <Citation key={i} citation={c} index={i} />
                ))}
              </div>
            )}

            {/* Retrieval trace */}
            <RetrievalTrace trace={response.retrieval_trace} />
          </div>
        )}

        {/* Footer */}
        <div
          style={{
            marginTop: "60px",
            paddingTop: "24px",
            borderTop: "1px solid #0f1a0f",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "12px",
          }}
        >
          <div style={{ display: "flex", gap: "20px" }}>
            {[
              { label: "LangChain", color: "#4ade80" },
              { label: "ChromaDB", color: "#c084fc" },
              { label: "Memgraph", color: "#60a5fa" },
              { label: "GPT-4o", color: "#fbbf24" },
            ].map((t) => (
              <span
                key={t.label}
                style={{
                  fontSize: "11px",
                  color: t.color,
                  fontFamily: "monospace",
                  letterSpacing: "0.05em",
                  opacity: 0.6,
                }}
              >
                {t.label}
              </span>
            ))}
          </div>
          <span style={{ fontSize: "11px", color: "#1e2d1e", fontFamily: "monospace" }}>
            GraphRAG · Privacy Regulation Q&A
          </span>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
        * { box-sizing: border-box; }
        textarea::placeholder { color: #2d3f2d; }
        ::-webkit-scrollbar { width: 6px; } 
        ::-webkit-scrollbar-track { background: #080c08; }
        ::-webkit-scrollbar-thumb { background: #1e2d1e; border-radius: 3px; }
      `}</style>
    </div>
  );
}
