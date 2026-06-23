"""Pipeline diagram HTML component for the ATS CV Scorer UI.

Architecture:
- get_pipeline_html()  → renders the diagram ONCE (yield 1).
- get_stage_signal()   → tiny hidden HTML updated at each subsequent yield.
- BRIDGE_JS            → MutationObserver injected via gr.Blocks(js=...).
                         Observes the signal component and updates the
                         persistent diagram DOM in-place — zero innerHTML
                         replacement, zero flickering, real CSS transitions.
"""

_GREEN = "#22c55e"
_GRAY = "#e0dfd8"
_HIDDEN = '<div id="ats-pipeline" style="display:none"></div>'

_CSS = """<style>
@keyframes borderPulse {
  0%,100% { border-color: #6366f1; }
  50%      { border-color: #a5b4fc; }
}
@keyframes spinDot { to { transform: rotate(360deg); } }
@keyframes slideDown {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes ats-draw { to { stroke-dashoffset: 0; } }
#ats-pipeline {
  background: var(--background-fill-primary, #fff);
  border: 0.5px solid var(--border-color-primary, #e0dfd8);
  border-radius: 10px;
  padding: 12px 14px;
  animation: slideDown .25s ease;
  font-family: Inter, sans-serif;
}
.ats-hdr {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}
.ats-lbl {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--body-text-color-subdued, #888780);
}
.ats-status {
  font-size: 11px;
  font-weight: 500;
  color: #6366f1;
  margin-left: 4px;
}
.ats-inner {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.ats-node {
  border: 1px solid #e0dfd8;
  border-radius: 8px;
  padding: 7px 16px;
  background: #fff;
  text-align: center;
  position: relative;
  transition: border-color .4s, background .4s;
}
.ats-node .nt {
  font-size: 11px;
  font-weight: 600;
  color: #888780;
  white-space: nowrap;
  transition: color .4s;
}
.ats-node .ns {
  font-size: 9px;
  color: #b4b2a9;
  margin-top: 1px;
  white-space: nowrap;
  transition: color .4s;
}
.ats-node.active {
  border-color: #6366f1;
  background: #eef2ff;
  animation: borderPulse 1.4s ease-in-out infinite;
}
.ats-node.active .nt { color: #4338ca; }
.ats-node.active .ns { color: #818cf8; }
.ats-node.done {
  border-color: #22c55e;
  background: #f0fdf4;
  animation: none;
}
.ats-node.done .nt { color: #15803d; }
.ats-node.done .ns { color: #86efac; }
.ats-sdot {
  display: none;
  position: absolute;
  top: -5px; right: -5px;
  width: 14px; height: 14px;
  border: 2px solid #6366f1;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spinDot .8s linear infinite;
}
.ats-node.active .ats-sdot { display: block; }
.ats-ck {
  display: none;
  position: absolute;
  top: -5px; right: -5px;
  width: 14px; height: 14px;
  background: #22c55e;
  border-radius: 50%;
  color: #fff;
  font-size: 8px;
  line-height: 14px;
  text-align: center;
}
.ats-node.done .ats-ck { display: block; }
.ats-fork-row {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  justify-content: center;
}
.dark .ats-node            { background: #1e1e2e; border-color: #3a3a5c; }
.dark .ats-node .nt        { color: #6b7280; }
.dark .ats-node.active     { background: #1e1b4b; border-color: #818cf8; }
.dark .ats-node.active .nt { color: #a5b4fc; }
.dark .ats-node.done       { background: #052e16; border-color: #4ade80; }
.dark .ats-node.done .nt   { color: #86efac; }
</style>"""

# JS injected once via gr.Blocks(js=BRIDGE_JS).
# Observes #ats-stage-signal for data-stage changes and updates the
# persistent diagram DOM in-place — no innerHTML replacement.
BRIDGE_JS = """
(function() {
  var G = '#22c55e', GR = '#e0dfd8';
  var IDX = {extract:0, nlp:1, sector:2, quality:3, done:4};

  var BLOCKS = {
    extract: {
      done:   ['ats-b0'],
      active: ['ats-b1']
    },
    nlp: {
      done:   ['ats-b0','ats-b1'],
      active: ['ats-b2']
    },
    sector: {
      done:   ['ats-b0','ats-b1','ats-b2'],
      active: ['ats-b3a','ats-b3b']
    },
    quality: {
      done:   ['ats-b0','ats-b1','ats-b2','ats-b3a','ats-b3b'],
      active: ['ats-b4']
    },
    done: {
      done:   ['ats-b0','ats-b1','ats-b2',
               'ats-b3a','ats-b3b','ats-b4','ats-b5'],
      active: []
    }
  };

  var ALL = [
    'ats-b0','ats-b1','ats-b2',
    'ats-b3a','ats-b3b','ats-b4','ats-b5'
  ];

  var STATUS = {
    extract: 'Lecture du texte en cours…',
    nlp:     'Compréhension du profil…',
    sector:  'Détection du secteur et des compétences…',
    quality: 'Calcul du score et recommandations…',
    done:    '✓ Analyse terminée'
  };

  // Simple connectors: {id, greenFrom (idx), newAt (idx)}
  var CV = [
    {id:'ats-cv01', gf:0, na:0},
    {id:'ats-cv12', gf:1, na:1},
    {id:'ats-cv45', gf:4, na:4}
  ];

  // Fork-in lines: {id, greenFrom, len, delay, dur}
  var FI = [
    {id:'ats-fi-vc', gf:2, len:12,  dl:0,   dr:0.3},
    {id:'ats-fi-h',  gf:2, len:160, dl:0.3, dr:1.4},
    {id:'ats-fi-vl', gf:2, len:12,  dl:1.7, dr:0.3},
    {id:'ats-fi-vr', gf:2, len:12,  dl:1.7, dr:0.3}
  ];

  // Fork-out lines
  var FO = [
    {id:'ats-fo-vl', gf:3, len:12,  dl:0,   dr:0.3},
    {id:'ats-fo-vr', gf:3, len:12,  dl:0,   dr:0.3},
    {id:'ats-fo-h',  gf:3, len:160, dl:0.3, dr:1.4},
    {id:'ats-fo-vc', gf:3, len:12,  dl:1.7, dr:0.3}
  ];

  function ge(id) { return document.getElementById(id); }

  function drawLine(e, len, dl, dr) {
    e.setAttribute('stroke-dasharray', len);
    e.setAttribute('stroke-dashoffset', len);
    e.style.animation = 'ats-draw '+dr+'s ease '+dl+'s forwards';
  }

  function clearLine(e) {
    e.removeAttribute('stroke-dasharray');
    e.removeAttribute('stroke-dashoffset');
    e.style.animation = '';
  }

  function applyStage(stage, retry) {
    if (!BLOCKS[stage]) return;
    // Diagram may not yet be in DOM if signal fires before HTML render.
    if (!ge('ats-b0')) {
      if ((retry || 0) < 10) {
        setTimeout(function() { applyStage(stage, (retry||0)+1); }, 50);
      }
      return;
    }

    var idx = IDX[stage];
    var st  = BLOCKS[stage];

    // Update block classes (CSS transitions play on persistent elements)
    ALL.forEach(function(id) {
      var e = ge(id); if (!e) return;
      e.classList.remove('active', 'done');
      if (st.done.indexOf(id) !== -1)   e.classList.add('done');
      else if (st.active.indexOf(id) !== -1) e.classList.add('active');
    });

    // Status text + header label
    var s = ge('ats-status');
    if (s) {
      s.textContent = STATUS[stage] || '';
      s.style.color = stage === 'done' ? G : '';
    }
    var lbl = document.querySelector('#ats-pipeline .ats-lbl');
    if (lbl) {
      lbl.textContent = stage === 'done'
        ? 'Analyse complète'
        : 'Analyse en cours';
    }

    // Simple vertical connectors
    CV.forEach(function(c) {
      var e = ge(c.id); if (!e) return;
      var green = idx >= c.gf;
      e.setAttribute('stroke', green ? G : GR);
      if (green && idx === c.na) drawLine(e, 14, 0, 2);
      else                        clearLine(e);
    });

    // Fork lines (in + out)
    FI.concat(FO).forEach(function(c) {
      var e = ge(c.id); if (!e) return;
      var green = idx >= c.gf;
      e.setAttribute('stroke', green ? G : GR);
      if (green && idx === c.gf) drawLine(e, c.len, c.dl, c.dr);
      else                        clearLine(e);
    });
  }

  // Watch #ats-stage-signal for data-stage changes
  function setupBridge() {
    var wrap = ge('ats-stage-signal');
    if (!wrap) { setTimeout(setupBridge, 200); return; }
    var obs = new MutationObserver(function() {
      var sig = wrap.querySelector('[data-stage]');
      if (sig) applyStage(sig.getAttribute('data-stage'));
    });
    obs.observe(wrap, {childList: true, subtree: true});
  }
  setupBridge();
})();
"""


def _node(nid: str, title: str, sub: str, min_w: int, extra_cls: str) -> str:
    cls = f"ats-node{extra_cls}"
    return (
        f'<div class="{cls}" id="{nid}" style="min-width:{min_w}px">'
        '<div class="ats-sdot"></div><div class="ats-ck">✓</div>'
        f'<div class="nt">{title}</div>'
        f'<div class="ns">{sub}</div>'
        "</div>"
    )


def _vline(lid: str) -> str:
    return (
        '<svg width="2" height="14" '
        'style="display:block;margin:0 auto;overflow:visible">'
        f'<line id="{lid}" x1="1" y1="0" x2="1" y2="14" '
        f'stroke="{_GRAY}" stroke-width="1.5" stroke-linecap="round"/>'
        "</svg>"
    )


def _fork_in() -> str:
    a = f'stroke="{_GRAY}" stroke-width="1.5" stroke-linecap="round"'
    return (
        '<svg width="300" height="24" '
        'style="display:block;margin:0 auto;overflow:visible">'
        f'<line id="ats-fi-vc" x1="150" y1="0"  x2="150" y2="12" {a}/>'
        f'<line id="ats-fi-h"  x1="70"  y1="12" x2="230" y2="12" {a}/>'
        f'<line id="ats-fi-vl" x1="70"  y1="12" x2="70"  y2="24" {a}/>'
        f'<line id="ats-fi-vr" x1="230" y1="12" x2="230" y2="24" {a}/>'
        "</svg>"
    )


def _fork_out() -> str:
    a = f'stroke="{_GRAY}" stroke-width="1.5" stroke-linecap="round"'
    return (
        '<svg width="300" height="24" '
        'style="display:block;margin:0 auto;overflow:visible">'
        f'<line id="ats-fo-vl" x1="70"  y1="0"  x2="70"  y2="12" {a}/>'
        f'<line id="ats-fo-vr" x1="230" y1="0"  x2="230" y2="12" {a}/>'
        f'<line id="ats-fo-h"  x1="70"  y1="12" x2="230" y2="12" {a}/>'
        f'<line id="ats-fo-vc" x1="150" y1="12" x2="150" y2="24" {a}/>'
        "</svg>"
    )


def get_pipeline_html(active_stage: str | None = None) -> str:
    """Return the initial pipeline diagram HTML (rendered once).

    None → hidden placeholder.
    Any other value → full diagram, b0 done + b1 active, all connectors
    gray. The JS bridge (BRIDGE_JS) handles all subsequent transitions
    in-place without replacing this HTML.
    """
    if active_stage is None:
        return _HIDDEN

    b0 = _node("ats-b0", "Votre CV est reçu", "fichier PDF déposé", 200, " done")
    b1 = _node(
        "ats-b1",
        "Lecture et extraction du texte",
        "lecture directe → reconnaissance optique → lecture visuelle IA",
        280,
        " active",
    )
    b2 = _node(
        "ats-b2",
        "Compréhension du profil",
        "poste · lieu · compétences · expériences identifiés",
        280,
        "",
    )
    b3a = _node("ats-b3a", "Détection du secteur", "métier identifié parmi 134", 140, "")
    b3b = _node(
        "ats-b3b",
        "Analyse des compétences",
        "comparaison avec les offres du marché",
        150,
        "",
    )
    b4 = _node(
        "ats-b4",
        "Score et recommandations",
        "compatibilité ATS · points forts · actions prioritaires",
        280,
        "",
    )
    b5 = _node("ats-b5", "Résultats affichés", "votre analyse est prête", 200, "")

    hdr = (
        '<div class="ats-hdr">'
        '<span class="ats-lbl">Analyse en cours</span>'
        '<span class="ats-status" id="ats-status">'
        "Lecture du texte en cours…"
        "</span>"
        "</div>"
    )
    fork_row = f'<div class="ats-fork-row">{b3a}{b3b}</div>'
    inner = (
        '<div class="ats-inner">'
        f"{b0}"
        f'{_vline("ats-cv01")}'
        f"{b1}"
        f'{_vline("ats-cv12")}'
        f"{b2}"
        f"{_fork_in()}"
        f"{fork_row}"
        f"{_fork_out()}"
        f"{b4}"
        f'{_vline("ats-cv45")}'
        f"{b5}"
        "</div>"
    )
    return f"{_CSS}<div id=\"ats-pipeline\">{hdr}{inner}</div>"


def get_stage_signal(stage: str) -> str:
    """Return the tiny signal HTML consumed by the JS bridge."""
    return f'<span data-stage="{stage}"></span>'
