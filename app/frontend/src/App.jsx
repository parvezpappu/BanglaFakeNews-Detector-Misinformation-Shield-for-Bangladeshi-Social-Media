import { useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const SAMPLE_INPUT = {
  category: "National",
  headline: "সামাজিক মাধ্যমে ছড়িয়ে পড়া দাবিটি নিয়ে নতুন আলোচনা",
  content:
    "সাম্প্রতিক সময়ে একটি খবর দ্রুত ছড়িয়ে পড়েছে। অনেকে এটিকে সত্য দাবি করলেও অন্যরা বলছেন তথ্যটি বিভ্রান্তিকর।"
};

const categories = [
  "National",
  "International",
  "Politics",
  "Sports",
  "Entertainment",
  "Finance",
  "Education",
  "Editorial",
  "Miscellaneous",
  "Lifestyle",
  "Technology",
  "Crime"
];

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
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

export default function App() {
  const [form, setForm] = useState(SAMPLE_INPUT);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form)
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Prediction failed.");
      }

      const payload = await response.json();
      setResult(payload);
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
  }

  return (
    <div className="app-shell">
      <div className="hero-backdrop" />
      <main className="layout">
        <section className="intro-panel">
          <p className="eyebrow">Bangla Fake News Detection</p>
          <h1>Ensemble-powered Bengali fact screening for live demos.</h1>
          <p className="lead">
            This interface sends Bengali news text to a FastAPI backend powered by our
            BanglaBERT + XGBoost ensemble and returns a real-time prediction with
            confidence breakdowns.
          </p>
          <div className="status-strip">
            <span>Model: BanglaBERT + XGBoost</span>
            <span>Mode: Real-time inference</span>
          </div>
        </section>

        <section className="workspace">
          <form className="input-card" onSubmit={handleSubmit}>
            <div className="card-head">
              <h2>Analyze News Text</h2>
              <button type="button" className="ghost-button" onClick={handleSampleLoad}>
                Load Sample
              </button>
            </div>

            <label className="field">
              <span>Category</span>
              <select value={form.category} onChange={(event) => updateField("category", event.target.value)}>
                {categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </label>

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
                  <BranchCard title="BanglaBERT Branch" probabilities={result.branch_probabilities.banglabert} />
                  <BranchCard title="XGBoost Branch" probabilities={result.branch_probabilities.xgboost} />
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p>The final prediction, confidence score, and branch probabilities will appear here.</p>
              </div>
            )}
          </section>
        </section>
      </main>
    </div>
  );
}
