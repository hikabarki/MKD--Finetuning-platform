"""
ui/styles.py — CSS for the MKD Fine-Tuning Platform Gradio UI.
"""

CSS = """
footer { display:none !important; }
* { box-sizing: border-box; }

/* ── Typography ───────────────────────────────────────────────────────────── */
body, .gradio-container, .gradio-container * {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
               sans-serif !important;
  -webkit-font-smoothing: antialiased !important;
}

/* ── Inputs & Textareas ───────────────────────────────────────────────────── */
input:not([type=range]):not([type=checkbox]):not([type=radio]),
textarea, select {
  color: #e6edf3 !important;
  background-color: #161b22 !important;
  border: 1px solid #30363d !important;
  border-radius: 6px !important;
  caret-color: #e6edf3 !important;
  font-size: 13px !important;
}
input:not([type=range])::placeholder, textarea::placeholder {
  color: #6e7681 !important;
  opacity: 1 !important;
}
input:not([type=range]):focus, textarea:focus {
  border-color: #58a6ff !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important;
}
/* Gradio svelte wrappers */
[class*="svelte"] input:not([type=range]):not([type=checkbox]):not([type=radio]),
[class*="svelte"] textarea,
.block input:not([type=range]):not([type=checkbox]):not([type=radio]),
.block textarea, .form input:not([type=range]):not([type=checkbox]):not([type=radio]),
.form textarea {
  color: #e6edf3 !important;
  background-color: #161b22 !important;
}

/* ── Labels ───────────────────────────────────────────────────────────────── */
.block label span, .block > label, .form > label,
span.text-sm, label > span, .label-wrap span,
.block label > span:first-child {
  color: #c9d1d9 !important;
  font-size: 13px !important;
  font-weight: 500 !important;
}
.block .info, .block span.text-xs { color: #8b949e !important; font-size: 11.5px !important; }

/* ── Buttons ──────────────────────────────────────────────────────────────── */
button { border-radius: 6px !important; font-weight: 500 !important;
         transition: background 0.15s, border-color 0.15s !important; }
button.primary, button[data-testid="primary"] {
  background: #238636 !important; border: 1px solid #238636 !important;
  color: #ffffff !important; font-weight: 600 !important;
}
button.primary:hover { background: #2ea043 !important; border-color: #2ea043 !important; }
button.secondary, button[data-testid="secondary"] {
  background: #21262d !important; border: 1px solid #30363d !important;
  color: #c9d1d9 !important;
}
button.secondary:hover { background: #30363d !important; }
button.stop, button[data-testid="stop"] {
  background: #3d1a1a !important; border: 1px solid #f85149 !important;
  color: #f85149 !important;
}
#land_dev_btn {
  background: #0d1b2a !important; border: 1px solid #58a6ff !important;
  color: #58a6ff !important; font-weight: 600 !important;
}
#land_dev_btn:hover {
  background: #1a2d42 !important; border-color: #79c0ff !important;
  color: #79c0ff !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.tab-nav { border-bottom: 1px solid #30363d !important; }
.tab-nav button {
  color: #8b949e !important; background: transparent !important;
  border-bottom: 2px solid transparent !important;
  font-size: 13px !important; padding: 8px 16px !important;
}
.tab-nav button.selected {
  color: #e6edf3 !important; border-bottom-color: #58a6ff !important;
  background: transparent !important; font-weight: 600 !important;
}
.tab-nav button:hover { color: #c9d1d9 !important; }

/* ── Dropdowns ────────────────────────────────────────────────────────────── */
ul[role="listbox"], ul.options, .dropdown-menu,
[data-testid="dropdown-options"], li[role="option"] {
  background: #1c2128 !important; color: #e6edf3 !important;
  border: 1px solid #30363d !important;
}
li[role="option"] {
  padding: 8px 14px !important; font-size: 13px !important;
}
li[role="option"]:hover, li[role="option"].selected {
  background: #2d333b !important; color: #e6edf3 !important;
}
li[role="option"] * { color: #e6edf3 !important; }
select option { background: #1c2128 !important; color: #e6edf3 !important; }

/* ── Popovers / Overflow menus ────────────────────────────────────────────── */
[popover], [popover] *, [role="menu"], [role="menu"] *,
[class*="overflow"], div[style*="position: absolute"],
div[style*="position:absolute"] {
  background-color: #1c2128 !important;
  color: #e6edf3 !important; border-color: #30363d !important;
}
button[aria-haspopup] { color: #8b949e !important; }

/* ── DataFrame / Table ────────────────────────────────────────────────────── */
.dataframe, .table-wrap, [data-testid="dataframe"] {
  background: #0d1117 !important;
  border: 1px solid #30363d !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
table { border-collapse: collapse !important; width: 100% !important; }
thead tr { background: #1c2128 !important; }
thead tr th {
  color: #c9d1d9 !important;
  font-size: 11.5px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  padding: 11px 16px !important;
  border-bottom: 2px solid #30363d !important;
  background: #1c2128 !important;
  white-space: nowrap !important;
}
tbody tr td {
  color: #e6edf3 !important;
  font-size: 13px !important;
  padding: 10px 16px !important;
  border-bottom: 1px solid #21262d !important;
  background: transparent !important;
}
tbody tr:nth-child(even) td { background: rgba(22,27,34,0.6) !important; }
tbody tr:hover td { background: #21262d !important; }

/* ── Checkboxes & Radios ──────────────────────────────────────────────────── */
input[type=checkbox], input[type=radio] {
  accent-color: #238636 !important;
  width: 15px !important; height: 15px !important;
}
.block label:has(input[type=checkbox]) span,
.block label:has(input[type=radio]) span {
  color: #c9d1d9 !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────────── */
input[type=range] { accent-color: #58a6ff !important; }
input[type=range]::-webkit-slider-thumb {
  background: #58a6ff !important; width: 16px !important; height: 16px !important;
}
input[type=range]::-webkit-slider-runnable-track {
  background: #30363d !important; height: 4px !important; border-radius: 2px !important;
}

/* ── Number & Password inputs (explicit overrides) ────────────────────────── */
input[type=number], input[type=password] {
  color: #e6edf3 !important;
  background-color: #161b22 !important;
  border: 1px solid #30363d !important;
  -webkit-text-fill-color: #e6edf3 !important;
}
input[type=password]::-webkit-credentials-auto-fill-button { display: none !important; }
input[type=password]:-webkit-autofill,
input[type=password]:-webkit-autofill:focus {
  -webkit-box-shadow: 0 0 0 1000px #161b22 inset !important;
  -webkit-text-fill-color: #e6edf3 !important;
}

/* ── Plot / Chart containers ─────────────────────────────────────────────── */
.gradio-plot, .plot-container, [data-testid="plot"],
.block:has(.gradio-plot), .block:has(canvas) {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.gradio-plot img, .plot-container img, [data-testid="plot"] img,
.gradio-plot canvas, .plot-container canvas {
  max-width: 100% !important;
  width: 100% !important;
  height: auto !important;
  display: block !important;
  border-radius: 8px !important;
}

/* ── Blocks ───────────────────────────────────────────────────────────────── */
.block { border-color: #30363d !important; border-radius: 8px !important; }

/* ── Scrollbars ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ── Code ─────────────────────────────────────────────────────────────────── */
.cm-content, .cm-line { color: #e6edf3 !important; }
pre, code {
  background: #161b22 !important; color: #e6edf3 !important;
  border-radius: 4px !important; border: 1px solid #30363d !important;
}

/* ── Chat bubbles ─────────────────────────────────────────────────────────── */
.message-bubble-border { border-color: #30363d !important; }
.bot, .bot * { background: #161b22 !important; color: #e6edf3 !important; }
.user, .user * { background: #1f6feb !important; color: #ffffff !important; }
[data-testid="bot"] { background: #161b22 !important; color: #e6edf3 !important; }
[data-testid="user"] { background: #1f6feb !important; color: #ffffff !important; }
.message { color: #e6edf3 !important; }
.message p, .message span, .message div,
.prose p, .prose span, .prose li { color: #e6edf3 !important; }

/* ── Accordion ────────────────────────────────────────────────────────────── */
.accordion-header { color: #c9d1d9 !important; background: #161b22 !important;
                    border-color: #30363d !important; }

/* ── File upload ──────────────────────────────────────────────────────────── */
.upload-container, [data-testid="upload"] {
  border: 2px dashed #30363d !important; background: #161b22 !important;
  border-radius: 8px !important;
}
[data-testid="upload"]:hover { border-color: #58a6ff !important; }
"""
