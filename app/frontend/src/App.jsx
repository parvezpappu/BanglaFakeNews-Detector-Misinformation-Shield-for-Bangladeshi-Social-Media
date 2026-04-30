import { useEffect, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const SAMPLE_INPUT = {
  headline: "তামিম সভাপতি নির্বাচিত হলে নিশ্চিতভাবেই উপকৃত হবে বাংলাদেশের ক্রিকেট, সাকিবের আশা",
  content:
    "নির্বাচনের মাধ্যমে পূর্ণ মেয়াদে বাংলাদেশ ক্রিকেট বোর্ডের (বিসিবি) সভাপতি হলে তামিম ইকবালের নেতৃত্বে দেশের ক্রিকেট নিশ্চিতভাবেই উপকৃত হবে বলে বুধবার আত্মবিশ্বাস ব্যক্ত করেছেন সাবেক অধিনায়ক সাকিব আল হাসান।গত বছরের ৬ অক্টোবর অনুষ্ঠিত বিসিবি নির্বাচনে ব্যাপক অনিয়মের অভিযোগে আমিনুল ইসলাম বুলবুলের নেতৃত্বাধীন বোর্ড ভেঙে দেয় জাতীয় ক্রীড়া পরিষদ (এনএসসি)। এরপর গত ৭ এপ্রিল বিসিবির অ্যাডহক কমিটির সভাপতি হিসেবে তামিমকে নিয়োগ দেওয়া হয়।এনএসসি তামিমের নেতৃত্বে ১১ সদস্যের একটি কমিটি গঠন করে দেয়, যাদের মূল কাজ হলো ৬ জুলাইয়ের মধ্যে নতুন নির্বাচনের আয়োজন করা। তবে বর্তমান বোর্ড আরও আগেই প্রক্রিয়াটি শেষ করতে চাইছে এবং সম্ভবত জুনের শুরুতেই নির্বাচন অনুষ্ঠিত হবে। সাবেক অধিনায়ক তামিম ইতোমধ্যে পূর্ণ ৪ বছরের মেয়াদের জন্য সভাপতি পদে প্রতিদ্বন্দ্বিতা করার ইচ্ছা প্রকাশ করেছেন।মুম্বাইয়ে 'ইইউ টি-টোয়েন্টি বেলজিয়াম ২০২৬'-এর জার্সি উন্মোচন অনুষ্ঠানের ফাঁকে গণমাধ্যমকে সাকিব বলেন, 'আমি বলতে চাই, তিনি (তামিম) তো নির্বাচিত হয়ে আসেননি। তিনি নির্বাচন আয়োজনের জন্য সেখানে দায়িত্ব পেয়েছেন এবং আশা করি, তিনি যদি সভাপতি হন, তার একটি দীর্ঘমেয়াদী পরিকল্পনা থাকবে এবং নিশ্চিতভাবেই বাংলাদেশ ক্রিকেট তার কাছ থেকে উপকৃত হবে।'সবশেষ আইসিসি টি-টোয়েন্টি বিশ্বকাপে বাংলাদেশের অংশ না নেওয়াকে পূর্ববর্তী সরকারের একটি বিরাট ভুল হিসেবে অভিহিত করেছেন সাকিব। তিনি এটিকে দেশের ক্রিকেটের জন্য বড় ধরনের ধাক্কা বলেও উল্লেখ করেছেন।চলতি বছরের শুরুতে ভারত ক্রিকেট বোর্ড (বিসিসিআই) কলকাতা নাইট রাইডার্সকে তাদের আইপিএল স্কোয়াড থেকে মোস্তাফিজুর রহমানকে ছেড়ে দেওয়ার নির্দেশ দেয়। এরপর নিরাপত্তা ইস্যুতে তৎকালীন অন্তর্বর্তীকালীন সরকার বাংলাদেশ জাতীয় দলকে ভারতে খেলতে যাওয়ার অনুমতি দেয়নি। পরবর্তীতে স্কটল্যান্ড বিশ্বকাপে বাংলাদেশের স্থলাভিষিক্ত হয়।সাকিব বলেন, 'আমি মনে করি, বাংলাদেশের ক্রিকেটের দৃষ্টিকোণ থেকে এটি একটি বড় ক্ষতি, বিরাট এক মিস। কারণ, একটি দেশ হিসেবে আমরা আমাদের খেলোয়াড়দের বিশ্বকাপ কাপ খেলতে দেখতে ভালোবাসি। বাংলাদেশের মতো একটি দেশ বিশ্বকাপে অংশগ্রহণ না করাটা অনেক বড় একটা মিস ছিল... এটি সরকারের পক্ষ থেকে একটা বড় ভুল ছিল যে, তারা সেই বিশ্বকাপে অংশগ্রহণ না করার সিদ্ধান্ত নিয়েছিল।'তবে সেই ধাক্কা কাটিয়ে দলের বর্তমান গতিপথ নিয়ে ইতিবাচক ধারণা পোষণ করেছেন বাঁহাতি অলরাউন্ডার সাকিব। বিশেষ করে, নিউজিল্যান্ডের বিপক্ষে চলমান সাদা বলের সিরিজের সাম্প্রতিক সাফল্যের পর তিনি এমন মন্তব্য করেছেন।তিনি বলেন, 'বাংলাদেশ দল এই মুহূর্তে সত্যিই ভালো করছে। তারা মাত্রই নিউজিল্যান্ডকে ওয়ানডে সিরিজে হারিয়েছে, তারা নিউজিল্যান্ডের বিপক্ষে প্রথম টি-টোয়েন্টিও জিতেছে। তাই ভালো দল আছে। আপনি জানেন, আগে এটি ব্যক্তিগত নির্ভর ছিল। কিন্তু এখন এটি মূলত দলকেন্দ্রিক এবং আমি মনে করি, এগিয়ে যাওয়ার জন্য এটাই ভালো উপায়।'আইপিএলে নিজের অনুপস্থিতি নিয়ে এক প্রশ্নের জবাবে সাকিব ইঙ্গিত দেন যে, ক্রিকেটে এখন নতুন প্রজন্মের জয়গান চলছে। তিনি বলেন, 'আমি যথেষ্ট খেলেছি আইপিএলে। নতুন প্রজন্ম এসেছে, তারা সত্যিই ভালো পারফর্ম করছে। কারণ, আমি আমার বয়সী খুব কম খেলোয়াড়কেই আইপিএল খেলতে দেখি। আমার মনে হয়, এক বা দুইজন, হয়তো সর্বোচ্চ পাঁচজন হবে।'"
};

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatErrorDetail(detail, fallback) {
  if (!detail) {
    return fallback;
  }
  if (typeof detail === "string") {
    return detail;
  }
  if (detail.error) {
    return detail.error;
  }
  return JSON.stringify(detail);
}

function ProbabilityBar({ label, value, tone = "primary" }) {
  return (
    <div className="probability-row">
      <div className="probability-meta">
        <span>{label}</span>
        <strong>{formatPercent(value)}</strong>
      </div>
      <div className="probability-track">
        <div
          className={`probability-fill ${tone}`}
          style={{ width: `${Math.max(value * 100, 2)}%` }}
        />
      </div>
    </div>
  );
}

function BranchCard({ title, probabilities }) {
  return (
    <div className="branch-card">
      <h3>{title}</h3>
      <ProbabilityBar label="Fake" value={probabilities.fake} tone="danger" />
      <ProbabilityBar label="Real" value={probabilities.real} tone="success" />
    </div>
  );
}

function EvidencePanel({ evidence, onCheckEvidence, loading }) {
  const [expanded, setExpanded] = useState(false);
  const items = evidence?.items || [];
  const visibleItems = expanded ? items : items.slice(0, 2);
  const hiddenCount = Math.max(items.length - 2, 0);

  return (
    <div className="evidence-block">
      <div className="evidence-head">
        <h3>🔍 Check Real Sources</h3>
        <span className="evidence-badge">
          {evidence ? evidence.status : "ready"}
        </span>
      </div>

      {!evidence ? (
        <div>
          <p>Search trusted Bangladeshi news portals for this headline:</p>
          <div className="trusted-sources">
            {["Prothom Alo", "BDNews24", "Daily Star", "Dhaka Tribune", "Jugantor"].map(site => (
              <span key={site} className="source-tag">{site}</span>
            ))}
          </div>
          <button className="evidence-button" onClick={onCheckEvidence} disabled={loading}>
            {loading ? "🔎 Searching..." : "🔎 Check Evidence"}
          </button>
        </div>
      ) : (
        <div>
          {evidence.items && evidence.items.length > 0 ? (
            <div>
              <p>Found {evidence.items.length} matching articles:</p>
              <ul className="evidence-list">
                {visibleItems.map((item, idx) => (
                  <li key={idx} className="evidence-item">
                    <a href={item.link} target="_blank" rel="noreferrer">
                      <strong>{item.title}</strong>
                    </a>
                    <p className="evidence-snippet">{item.snippet}</p>
                    <span className="evidence-source">📰 {item.source}</span>
                  </li>
                ))}
              </ul>
              {hiddenCount > 0 ? (
                <button
                  type="button"
                  className="evidence-more-button"
                  onClick={() => setExpanded((current) => !current)}
                >
                  {expanded ? "Show less" : `Show ${hiddenCount} more ->`}
                </button>
              ) : null}
            </div>
          ) : (
            <p className="evidence-note">{evidence.note || "No results found."}</p>
          )}

          <a href={evidence.search_url} target="_blank" rel="noreferrer" className="manual-search-link">
            🔗 Open Full Google Search
          </a>

          <button
            className="evidence-button secondary"
            onClick={onCheckEvidence}
            disabled={loading}
          >
            {loading ? "⏳ Searching..." : "🔄 Refresh Search"}
          </button>
        </div>
      )}
    </div>
  );
}

function HistoryPanel({ history, loading, enabled, message, onRefresh, onDelete }) {
  return (
    <section className="history-panel">
      <div className="card-head">
        <h2>History</h2>
        <button type="button" className="ghost-button" onClick={onRefresh} disabled={loading}>
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {!enabled ? (
        <p className="history-note">
          MongoDB history is not connected{message ? `: ${message}` : "."}
        </p>
      ) : history.length === 0 ? (
        <p className="history-note">No tested news yet.</p>
      ) : (
        <ul className="history-list">
          {history.map((item) => (
            <li key={item.id} className="history-item">
              <div className="history-card-head">
                <strong>{item.headline || "Untitled news"}</strong>
                <span className={`history-label ${item.label === "fake" ? "fake" : "real"}`}>
                  {item.label.toUpperCase()}
                </span>
              </div>
              <p>{item.content}</p>
              <div className="history-meta">
                <span>{formatPercent(item.confidence)}</span>
                <span>{item.created_at ? new Date(item.created_at).toLocaleString() : ""}</span>
              </div>
              <button
                type="button"
                className="history-delete"
                onClick={() => onDelete(item.id)}
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default function App() {
  const [activeView, setActiveView] = useState("analyze");
  const [form, setForm] = useState(SAMPLE_INPUT);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [evidence, setEvidence] = useState(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [historyEnabled, setHistoryEnabled] = useState(true);
  const [historyMessage, setHistoryMessage] = useState("");
  const [historyLoading, setHistoryLoading] = useState(false);

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function loadHistory() {
    setHistoryLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/history?limit=10`);
      if (!response.ok) {
        throw new Error("History fetch failed");
      }
      const payload = await response.json();
      setHistoryEnabled(Boolean(payload.enabled));
      setHistoryMessage(payload.message || "");
      setHistory(payload.items || []);
    } catch {
      setHistoryEnabled(false);
      setHistoryMessage("Could not reach the history endpoint.");
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function deleteHistoryItem(id) {
    const confirmed = window.confirm("Delete this history item?");
    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/history/${id}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error("Delete failed");
      }
      setHistory((current) => current.filter((item) => item.id !== id));
    } catch {
      setHistoryMessage("Could not delete the selected history item.");
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  async function fetchEvidence() {
    setEvidenceLoading(true);
    setEvidence(null);

    try {
      const response = await fetch(`${API_BASE_URL}/check-evidence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          headline: form.headline,
          content: form.content,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Evidence search failed");
      }

      const payload = await response.json();
      setEvidence(payload);
    } catch (e) {
      setEvidence({
        status: "error",
        note: e.message || "Search service unavailable. Try manual search below.",
        search_url: `https://www.google.com/search?q=${encodeURIComponent(form.headline)}`,
        items: [],
      });
    } finally {
      setEvidenceLoading(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setEvidence(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          headline: form.headline,
          content: form.content,
          include_evidence: false,
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(formatErrorDetail(payload.detail, "Prediction failed."));
      }

      const payload = await response.json();
      setResult(payload);
      loadHistory();
      if (payload.label === "real") {
        fetchEvidence();
      }
    } catch (submitError) {
      setError(submitError.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function handleSampleLoad() {
    setForm(SAMPLE_INPUT);
    setError("");
    setResult(null);
    setEvidence(null);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <span className="brand-mark">BN</span>
          <div>
            <p>Bangla Fake News</p>
            <strong>Detection Console</strong>
          </div>
        </div>
        <nav className="side-nav" aria-label="Main sections">
          <button
            type="button"
            className={activeView === "analyze" ? "active" : ""}
            onClick={() => setActiveView("analyze")}
          >
            Analyze
          </button>
          <button
            type="button"
            className={activeView === "history" ? "active" : ""}
            onClick={() => {
              setActiveView("history");
              loadHistory();
            }}
          >
            History
          </button>
        </nav>
        <div className="side-stats">
          <span>BanglaBERT + LightGBM</span>
          <strong>93.6% accuracy</strong>
        </div>
      </aside>

      <main className="layout">
        <header className="topbar">
          <div>
            <p className="eyebrow">Bangla Fake News Detection</p>
            <h1>{activeView === "history" ? "Review History" : "Misinformation Shield"}</h1>
          </div>
          <div className="status-strip">
            <span>Model: BanglaBERT + LightGBM</span>
            <span>Accuracy: 93.6%</span>
          </div>
        </header>

        {activeView === "analyze" ? (
          <section className="workspace">
            <form className="input-card" onSubmit={handleSubmit}>
            <div className="card-head">
              <h2>Analyze News Text</h2>
              <button type="button" className="ghost-button" onClick={handleSampleLoad}>
                Load Sample
              </button>
            </div>

            <label className="field">
              <span>Headline</span>
              <textarea
                rows={3}
                value={form.headline}
                onChange={(event) => updateField("headline", event.target.value)}
                placeholder="বাংলা শিরোনাম লিখুন"
              />
            </label>

            <label className="field">
              <span>Content</span>
              <textarea
                rows={8}
                value={form.content}
                onChange={(event) => updateField("content", event.target.value)}
                placeholder="খবরের কনটেন্ট লিখুন"
              />
            </label>

            <button className="submit-button" type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Run Prediction"}
            </button>

            {error ? <p className="error-text">{error}</p> : null}
            </form>

            <section className="result-card">
            <div className="card-head">
              <h2>Prediction Output</h2>
              <span className="badge">{result ? "Ready" : "Waiting"}</span>
            </div>

            {result ? (
              <>
                <div className={`verdict ${result.label === "fake" ? "verdict-fake" : "verdict-real"}`}>
                  <div>
                    <p className="verdict-label">Ensemble Verdict</p>
                    <h3>{result.label.toUpperCase()}</h3>
                  </div>
                  <div className="confidence-pill">{formatPercent(result.confidence)}</div>
                </div>

                <div className="ensemble-block">
                  <h3>Final Ensemble Probabilities</h3>
                  <ProbabilityBar label="Fake" value={result.probabilities.fake} tone="danger" />
                  <ProbabilityBar label="Real" value={result.probabilities.real} tone="success" />
                </div>

                <div className="branch-grid">
                  <BranchCard title="BanglaBERT" probabilities={result.branch_probabilities.banglabert} />
                  <BranchCard title="LightGBM" probabilities={result.branch_probabilities.lightgbm} />
                </div>

                {result.label === "real" ? (
                  <EvidencePanel
                    evidence={evidence}
                    onCheckEvidence={fetchEvidence}
                    loading={evidenceLoading}
                  />
                ) : null}
              </>
            ) : (
              <div className="empty-state">
                <p>The prediction, confidence, and branch probabilities will appear here.</p>
              </div>
            )}
            </section>
          </section>
        ) : (
          <HistoryPanel
            history={history}
            loading={historyLoading}
            enabled={historyEnabled}
            message={historyMessage}
            onRefresh={loadHistory}
            onDelete={deleteHistoryItem}
          />
        )}
      </main>
    </div>
  );
}
