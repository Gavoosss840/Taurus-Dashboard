/* ── Taurus Dashboard — logique d'interface ─────────────────────────────
   Vanilla JavaScript, sans dépendance ni étape de compilation : le fichier
   est servi tel quel par FastAPI.                                          */

"use strict";

const el = (id) => document.getElementById(id);

const dom = {
  form:        el("search-form"),
  input:       el("ticker"),
  submit:      el("submit-button"),
  loading:     el("loading"),
  loadingText: el("loading-text"),
  error:       el("error"),
  errorMsg:    el("error-message"),
  result:      el("result"),
  methodPanel: el("method-panel"),
  methodToggle: el("method-toggle"),
  methodWeights: el("method-weights"),
  refresh:     el("refresh-button"),
};

let lastTicker = "";

/* ── Formatage ─────────────────────────────────────────────────────────── */

const NBSP = " ";   // espace fine insécable, séparateur de milliers

function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/d";
  return value.toLocaleString("fr-FR", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function formatPrice(value, currency) {
  if (value === null || value === undefined) return "n/d";
  const symbol = { USD: "$", EUR: "€", GBP: "£" }[currency] || "";
  return symbol + formatNumber(value, 2);
}

function formatPercent(value, decimals = 1, signed = true) {
  if (value === null || value === undefined) return "n/d";
  const sign = signed && value > 0 ? "+" : "";
  return `${sign}${formatNumber(value, decimals)}${NBSP}%`;
}

function formatAmount(value, currency) {
  if (value === null || value === undefined) return "n/d";
  const symbol = { USD: "$", EUR: "€", GBP: "£" }[currency] || "";
  const abs = Math.abs(value);
  if (abs >= 1e12) return `${symbol}${formatNumber(value / 1e12, 2)}${NBSP}Bn`;
  if (abs >= 1e9)  return `${symbol}${formatNumber(value / 1e9, 1)}${NBSP}Md`;
  if (abs >= 1e6)  return `${symbol}${formatNumber(value / 1e6, 1)}${NBSP}M`;
  return symbol + formatNumber(value, 0);
}

/* Classe CSS dérivée du verdict, utilisée pour la couleur. */
function verdictKey(verdict) {
  if (verdict === "SOUS-ÉVALUÉE") return "under";
  if (verdict === "SUR-ÉVALUÉE")  return "over";
  return "fair";
}

/* ── Bascule d'affichage ───────────────────────────────────────────────── */

function show(node)  { node.hidden = false; }
function hide(node)  { node.hidden = true; }

function setBusy(busy, ticker) {
  dom.submit.disabled = busy;
  dom.submit.textContent = busy ? "Analyse…" : "Analyser";
  if (busy) {
    dom.loadingText.textContent =
      `Analyse de ${ticker} — téléchargement des cours, des facteurs ` +
      `Fama-French et des comptes déposés à la SEC…`;
    show(dom.loading);
    hide(dom.error);
    hide(dom.result);
  } else {
    hide(dom.loading);
  }
}

function showError(message) {
  dom.errorMsg.textContent = message;
  show(dom.error);
  hide(dom.result);
}

/* ── Rendu du verdict ──────────────────────────────────────────────────── */

function renderVerdict(data) {
  const key = verdictKey(data.verdict);
  const card = el("verdict-card");
  card.dataset.verdict = key;

  el("company-name").textContent = data.company_name || data.ticker;
  el("ticker-badge").textContent = data.ticker;
  el("sector").textContent = translateSector(data.sector);
  el("verdict-label").textContent = data.verdict_label;
  el("verdict-score").textContent =
    `score composite ${data.composite_score >= 0 ? "+" : ""}${formatNumber(data.composite_score, 2)}`;
  el("summary").textContent = data.summary;

  /* La jauge couvre [-2, +2] : les bornes du score de chaque pilier. */
  const clamped = Math.max(-2, Math.min(2, data.composite_score ?? 0));
  el("gauge-needle").style.left = `${((clamped + 2) / 4) * 100}%`;

  el("metric-price").textContent = formatPrice(data.price, data.currency);
  el("metric-fair").textContent  = formatPrice(data.fair_value, data.currency);

  const upside = el("metric-upside");
  upside.textContent = formatPercent(data.upside_pct);
  upside.className = data.upside_pct === null ? ""
    : data.upside_pct > 0 ? "positive" : "negative";

  el("metric-mcap").textContent = formatAmount(data.market_cap, data.currency);
  el("metric-confidence").textContent =
    data.confidence === null ? "n/d" : `${Math.round(data.confidence * 100)}${NBSP}%`;
}

/* ── Rendu des piliers ─────────────────────────────────────────────────── */

function detailRows(pillar, currency) {
  const d = pillar.details || {};
  const rows = [];
  const pct = (v, n = 1) => v === null || v === undefined ? "n/d" : formatPercent(v * 100, n, false);

  if (pillar.key === "alpha") {
    rows.push(["Alpha annualisé", formatPercent(d.alpha_annual * 100)]);
    rows.push(["Statistique t", formatNumber(d.t_stat, 2)]);
    rows.push(["Seuil de significativité", `±${formatNumber(d.t_critical, 2)}`]);
    rows.push(["Valeur p", formatNumber(d.p_value, 3)]);
    rows.push(["R² de la régression", pct(d.r_squared, 0)]);
    rows.push(["Mois observés", d.n_obs ?? "n/d"]);
    rows.push(["Fenêtre", d.window || "n/d"]);
    if (d.betas) {
      const labels = {
        "Mkt-RF": "β marché", SMB: "β taille", HML: "β valeur",
        RMW: "β rentabilité", CMA: "β investissement",
      };
      for (const [factor, label] of Object.entries(labels)) {
        if (d.betas[factor] !== undefined) {
          rows.push([label, formatNumber(d.betas[factor], 2)]);
        }
      }
    }
  } else if (pillar.key === "capital_structure") {
    rows.push(["Divergence", formatPercent(d.divergence_pct)]);
    rows.push(["Seuil de déclenchement", `±${formatNumber(d.threshold_pct, 0)}${NBSP}%`]);
    rows.push(["Capitaux propres théoriques", formatAmount(d.fair_equity_value, currency)]);
    rows.push(["Capitalisation boursière", formatAmount(d.market_cap, currency)]);
    rows.push(["Résultat d'exploitation net d'impôt", formatAmount(d.nopat, currency)]);
    rows.push(["Valeur de la firme non endettée", formatAmount(d.unlevered_value, currency)]);
    rows.push(["Taux d'actualisation r_U", pct(d.discount_rate)]);
    rows.push(["Croissance perpétuelle g", pct(d.growth_rate)]);
    rows.push(["Bêta dé-leviérisé", formatNumber(d.unlevered_beta, 2)]);
    rows.push(["Bouclier fiscal de la dette", formatAmount(d.pv_tax_shield, currency)]);
    rows.push(["Coûts de détresse financière", formatAmount(d.pv_distress, currency)]);
    rows.push(["Coûts d'agence", formatAmount(d.pv_agency, currency)]);
    rows.push(["Dette nette", formatAmount(d.net_debt, currency)]);
    rows.push(["Probabilité de défaut à 1 an", pct(d.prob_default, 2)]);
    rows.push(["Spread de crédit estimé", pct(d.credit_spread, 2)]);
    rows.push(["Levier dette / capitaux propres", formatNumber(d.leverage_ratio, 2)]);
    rows.push(["Couverture des intérêts",
      d.interest_coverage === null ? "non applicable" : `${formatNumber(d.interest_coverage, 1)}×`]);
  } else if (pillar.key === "momentum") {
    rows.push(["Rendement cumulé 12-1", formatPercent(d.raw * 100)]);
    rows.push(["Volatilité annualisée", pct(d.volatility, 0)]);
    rows.push(["Momentum ajusté du risque", formatNumber(d.sharpe, 2)]);
    rows.push(["Même mesure pour le marché", formatNumber(d.market_sharpe, 2)]);
    rows.push(["Écart au marché", formatNumber(d.excess_sharpe, 2)]);
    rows.push(["Régime de krach détecté", d.crash_regime ? "oui" : "non"]);
    rows.push(["Fenêtre", d.window || "n/d"]);
  }
  return rows;
}

function renderSensitivity(grid, basePrice, currency, baseRate, baseGrowth) {
  if (!Array.isArray(grid) || grid.length === 0) return "";

  const rates = [...new Set(grid.map((c) => c.discount_rate))].sort((a, b) => a - b);
  const growths = [...new Set(grid.map((c) => c.growth_rate))].sort((a, b) => a - b);
  const lookup = new Map(grid.map((c) => [`${c.discount_rate}|${c.growth_rate}`, c]));

  let html = '<div class="sensitivity"><p class="sensitivity-title">'
    + 'Juste valeur par action selon le taux d\'actualisation et la croissance '
    + '<span class="sensitivity-hint">(encadré : scénario retenu)</span></p>'
    + '<table><thead><tr>'
    + `<th>r_U \\ g</th>`;
  for (const g of growths) {
    html += `<th>${formatNumber(g * 100, 1)}${NBSP}%</th>`;
  }
  html += "</tr></thead><tbody>";

  for (const r of rates) {
    html += `<tr><th>${formatNumber(r * 100, 1)}${NBSP}%</th>`;
    for (const g of growths) {
      const cell = lookup.get(`${r}|${g}`);
      if (!cell || cell.price_per_share === null) { html += "<td>n/d</td>"; continue; }
      const classes = [];
      if (basePrice && cell.price_per_share > basePrice) classes.push("cell-under");
      else if (basePrice) classes.push("cell-over");
      /* Encadre le scénario effectivement retenu pour le verdict. */
      const isBase = Math.abs(r - (baseRate ?? NaN)) < 1e-6
                  && Math.abs(g - (baseGrowth ?? NaN)) < 1e-6;
      if (isBase) classes.push("cell-base");
      html += `<td class="${classes.join(" ")}">${formatPrice(cell.price_per_share, currency)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table></div>";
  return html;
}

function renderPillars(data) {
  const container = el("pillars");
  container.innerHTML = "";

  for (const pillar of data.pillars) {
    const key = verdictKey(pillar.verdict);
    const article = document.createElement("article");
    article.className = "pillar" + (pillar.available ? "" : " unavailable");
    article.dataset.verdict = key;

    /* Barre de score centrée sur zéro, bornée à ±2. */
    const score = Math.max(-2, Math.min(2, pillar.score ?? 0));
    const half = Math.abs(score) / 2 * 50;
    const left = score >= 0 ? 50 : 50 - half;

    const rows = detailRows(pillar, data.currency);
    const tableRows = rows
      .map(([label, value]) => `<tr><th>${label}</th><td>${value}</td></tr>`)
      .join("");

    const sensitivity = pillar.key === "capital_structure"
      ? renderSensitivity(
          pillar.details?.sensitivity_grid, data.price, data.currency,
          pillar.details?.discount_rate, pillar.details?.growth_rate,
        )
      : "";

    article.innerHTML = `
      <div class="pillar-head">
        <h3 class="pillar-name">${pillar.name}</h3>
        <span class="pillar-weight">poids ${Math.round((pillar.weight ?? 0) * 100)}${NBSP}%</span>
      </div>
      <p class="pillar-headline">${pillar.headline}</p>
      <p class="pillar-score">score ${score >= 0 ? "+" : ""}${formatNumber(score, 2)} — ${pillar.verdict.toLowerCase()}</p>
      <div class="score-bar"><span class="score-fill" style="left:${left}%;width:${half}%"></span></div>
      <p class="pillar-explanation">${pillar.explanation}</p>
      ${rows.length ? `<details><summary>Détail du calcul</summary>
        <table class="detail-table"><tbody>${tableRows}</tbody></table>
        ${sensitivity}</details>` : ""}
    `;
    container.appendChild(article);
  }
}

function renderWarnings(data) {
  const panel = el("warnings");
  const list = el("warnings-list");
  list.innerHTML = "";
  if (!data.warnings || data.warnings.length === 0) { hide(panel); return; }
  for (const warning of data.warnings) {
    const item = document.createElement("li");
    item.textContent = warning;
    list.appendChild(item);
  }
  show(panel);
}

function renderProvenance(data) {
  const parts = Object.entries(data.data_sources || {})
    .map(([label, source]) => `${label} : ${source}`);
  el("sources").textContent =
    `Sources — ${parts.join(" · ")}. Calculé le ${data.computed_at} `
    + `en ${formatNumber(data.elapsed_seconds, 1)}${NBSP}s.`;
}

/* Le secteur vient de la nomenclature GICS, en anglais chez les fournisseurs. */
function translateSector(sector) {
  const map = {
    "Information Technology": "Technologies de l'information",
    "Communication Services": "Services de communication",
    "Health Care": "Santé",
    "Consumer Discretionary": "Consommation discrétionnaire",
    "Consumer Staples": "Consommation de base",
    "Industrials": "Industrie",
    "Materials": "Matériaux",
    "Energy": "Énergie",
    "Financials": "Finance",
    "Real Estate": "Immobilier",
    "Utilities": "Services aux collectivités",
    "Unknown": "Secteur indéterminé",
  };
  return map[sector] || sector;
}

/* ── Appel de l'API ────────────────────────────────────────────────────── */

async function runAnalysis(ticker, refresh = false) {
  const symbol = (ticker || "").trim().toUpperCase();
  if (!symbol) return;

  lastTicker = symbol;
  setBusy(true, symbol);

  try {
    const url = `/api/analyze/${encodeURIComponent(symbol)}${refresh ? "?refresh=true" : ""}`;
    const response = await fetch(url);
    const payload = await response.json();

    if (!response.ok) {
      showError(payload.error || payload.detail || `Erreur ${response.status}.`);
      return;
    }

    renderVerdict(payload);
    renderPillars(payload);
    renderWarnings(payload);
    renderProvenance(payload);
    show(dom.result);

    /* L'URL devient partageable et rejouable. */
    history.replaceState(null, "", `?ticker=${encodeURIComponent(symbol)}`);
    document.title = `${symbol} — ${payload.verdict_label} · Taurus Dashboard`;
  } catch (err) {
    showError(
      "Le serveur n'a pas répondu. Vérifiez qu'il est démarré "
      + "(./run.sh) puis réessayez."
    );
  } finally {
    setBusy(false);
  }
}

/* ── Volet méthode ─────────────────────────────────────────────────────── */

async function loadEngineConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) return;
    const cfg = await response.json();
    dom.methodWeights.textContent =
      `Pondération du score composite — alpha ${Math.round(cfg.weights.alpha * 100)}${NBSP}%, `
      + `structure du capital ${Math.round(cfg.weights.capital_structure * 100)}${NBSP}%, `
      + `momentum ${Math.round(cfg.weights.momentum * 100)}${NBSP}%. `
      + `Verdict au-delà de ±${formatNumber(cfg.verdict_threshold, 2)}, `
      + `«${NBSP}fortement${NBSP}» au-delà de ±${formatNumber(cfg.verdict_strong_threshold, 2)}. `
      + `Régression sur ${cfg.lookback_months} mois · momentum `
      + `${cfg.momentum_months}-${cfg.momentum_skip} · seuil Modigliani-Miller `
      + `±${formatNumber(cfg.leverage_gap_threshold_pct, 0)}${NBSP}% · taux sans risque `
      + `${formatNumber(cfg.risk_free_rate_annual * 100, 1)}${NBSP}% · prime de risque actions `
      + `${formatNumber(cfg.equity_risk_premium * 100, 1)}${NBSP}%.`;
  } catch (err) {
    /* Le volet méthode reste lisible sans ces chiffres. */
  }
}

/* ── Câblage ───────────────────────────────────────────────────────────── */

dom.form.addEventListener("submit", (event) => {
  event.preventDefault();
  runAnalysis(dom.input.value);
});

dom.refresh.addEventListener("click", () => runAnalysis(lastTicker, true));

dom.methodToggle.addEventListener("click", () => {
  const open = !dom.methodPanel.hidden;
  dom.methodPanel.hidden = open;
  dom.methodToggle.setAttribute("aria-expanded", String(!open));
});

for (const chip of document.querySelectorAll(".chip")) {
  chip.addEventListener("click", () => {
    dom.input.value = chip.dataset.ticker;
    runAnalysis(chip.dataset.ticker);
  });
}

/* Un ticker passé dans l'URL lance l'analyse au chargement. */
const initialTicker = new URLSearchParams(location.search).get("ticker");
if (initialTicker) {
  dom.input.value = initialTicker.toUpperCase();
  runAnalysis(initialTicker);
}

loadEngineConfig();
dom.input.focus();
