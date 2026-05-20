import { useEffect, useRef, useState, useCallback } from 'react';
import Scatterplot from 'deepscatter';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import './index.css';

/* ── Utilities ──────────────────────────────────────────── */

function generateColors(n) {
  const golden = 137.508;
  const colors = [];
  for (let i = 0; i < n; i++) {
    const hue = (i * golden) % 360;
    const sat = 70 + (i % 3) * 8;
    const lum = 60 + (i % 2) * 8;
    colors.push(`hsl(${hue}, ${sat}%, ${lum}%)`);
  }
  return colors;
}

function renderTextWithMath(text) {
  if (!text) return { __html: '' };
  let html = text;
  html = html.replace(/&lt;\/?title&gt;/gi, '');
  html = html.replace(/<\/?title>/gi, '');
  html = html.replace(/&lt;\/?jats:[a-z]+&gt;/gi, '');
  html = html.replace(/<\/?jats:[a-z]+>/gi, '');
  try {
    html = html.replace(/\$\$(.*?)\$\$/g, (_, m) =>
      katex.renderToString(m, { throwOnError: false, displayMode: true })
    );
    html = html.replace(/\$(.*?)\$/g, (match, m) => {
      if (m.match(/^\d/)) return match;
      return katex.renderToString(m, { throwOnError: false, displayMode: false });
    });
  } catch (e) { /* ignore */ }
  return { __html: html };
}

function sourceLabel(s) {
  return ({ openalex: 'OpenAlex', unam_repository: 'Repositorio UNAM', scielo_mexico: 'SciELO México' }[s] || s);
}

/* ── Viridis color scale for SOM ── */
const VIRIDIS = [
  [68,1,84],[72,35,116],[64,67,135],[52,94,141],[41,120,142],
  [32,144,140],[34,167,132],[53,183,121],[94,201,97],[144,214,67],
  [207,225,28],[253,231,37],
];

function viridisColor(t) {
  const idx = Math.min(Math.max(t, 0), 1) * (VIRIDIS.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, VIRIDIS.length - 1);
  const f = idx - lo;
  return [
    Math.round(VIRIDIS[lo][0] + f * (VIRIDIS[hi][0] - VIRIDIS[lo][0])),
    Math.round(VIRIDIS[lo][1] + f * (VIRIDIS[hi][1] - VIRIDIS[lo][1])),
    Math.round(VIRIDIS[lo][2] + f * (VIRIDIS[hi][2] - VIRIDIS[lo][2])),
  ];
}

/* ── SOM Heatmap Component ──────────────────────────────── */

function SomHeatmap({ somData, colors, clusterMeta, onPointHover, width, height }) {
  const canvasRef = useRef(null);
  const [hoveredCell, setHoveredCell] = useState(null);

  useEffect(() => {
    if (!somData || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const gx = somData.grid_x;
    const gy = somData.grid_y;
    const cellW = width / gy;
    const cellH = height / gx;

    canvas.width = width;
    canvas.height = height;

    // Find min/max for normalization
    let dMin = Infinity, dMax = -Infinity;
    for (const row of somData.umatrix) {
      for (const v of row) {
        if (v < dMin) dMin = v;
        if (v > dMax) dMax = v;
      }
    }
    const dRange = dMax - dMin || 1;

    // Draw U-Matrix cells
    for (let x = 0; x < gx; x++) {
      for (let y = 0; y < gy; y++) {
        const t = (somData.umatrix[x][y] - dMin) / dRange;
        const [r, g, b] = viridisColor(t);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(y * cellW, x * cellH, cellW + 1, cellH + 1);
      }
    }

    // Draw sampled points
    if (somData.points && colors.length) {
      for (const pt of somData.points) {
        const cx = pt.som_y * cellW + cellW / 2;
        const cy = pt.som_x * cellH + cellH / 2;
        const clusterColor = colors[pt.cluster] || '#888';

        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(2, cellW * 0.15), 0, Math.PI * 2);
        ctx.fillStyle = clusterColor;
        ctx.globalAlpha = 0.7;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }

    // Highlight hovered cell
    if (hoveredCell) {
      const { x, y } = hoveredCell;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.strokeRect(y * cellW, x * cellH, cellW, cellH);
    }
  }, [somData, colors, hoveredCell, width, height]);

  const handleMouseMove = useCallback((e) => {
    if (!somData) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const cellW = width / somData.grid_y;
    const cellH = height / somData.grid_x;
    const cellY = Math.floor(mx / cellW);
    const cellX = Math.floor(my / cellH);

    if (cellX >= 0 && cellX < somData.grid_x && cellY >= 0 && cellY < somData.grid_y) {
      setHoveredCell({ x: cellX, y: cellY });

      // Find a point in this cell to show in sidebar
      const pt = somData.points?.find(p => p.som_x === cellX && p.som_y === cellY);
      if (pt && onPointHover) {
        const density = somData.density?.[cellX]?.[cellY] || 0;
        const dist = somData.umatrix?.[cellX]?.[cellY] || 0;
        onPointHover({
          title: pt.title,
          year: pt.year,
          cluster: pt.cluster,
          faculty: `Cell (${cellX}, ${cellY}) · ${density} documents`,
          source: `U-Matrix distance: ${dist.toFixed(4)}`,
          abstract: null,
          url: null,
        });
      }
    }
  }, [somData, onPointHover, width, height]);

  return (
    <div className="som-container" style={{ width, height }}>
      <canvas
        ref={canvasRef}
        className="som-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredCell(null)}
      />
      {hoveredCell && somData && (
        <div
          className="som-tooltip"
          style={{
            left: hoveredCell.y * (width / somData.grid_y) + (width / somData.grid_y) / 2,
            top: hoveredCell.x * (height / somData.grid_x) - 28,
          }}
        >
          Cell ({hoveredCell.x}, {hoveredCell.y}) · {somData.density?.[hoveredCell.x]?.[hoveredCell.y] || 0} docs
        </div>
      )}
    </div>
  );
}

/* ── Main App ───────────────────────────────────────────── */

function App() {
  const scatterRef = useRef(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [clusterMeta, setClusterMeta] = useState([]);
  const [allLabels, setAllLabels] = useState([]);
  const [colors, setColors] = useState([]);
  const [activeCluster, setActiveCluster] = useState(null);
  const [labelPositions, setLabelPositions] = useState([]);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [activeView, setActiveView] = useState('atlas');
  const [somData, setSomData] = useState(null);

  // ── Load cluster metadata + labels ──
  useEffect(() => {
    fetch('http://localhost:8000/data/tiles/cluster_labels.json')
      .then(r => r.json())
      .then(data => {
        const clusters = data.clusters || [];
        setClusterMeta(clusters);
        setAllLabels(data.labels || []);
        setColors(generateColors(clusters.length));
      })
      .catch(() => console.warn('Could not load cluster labels'));
  }, []);

  // ── Load SOM data ──
  useEffect(() => {
    fetch('http://localhost:8000/data/tiles/som_umatrix.json')
      .then(r => r.json())
      .then(data => setSomData(data))
      .catch(() => console.warn('Could not load SOM data'));
  }, []);

  // ── Convert data coords → screen coords for floating labels ──
  const updateLabelPositions = useCallback(() => {
    const scatter = scatterRef.current;
    if (!scatter?._zoom || !allLabels.length) return;

    try {
      const scales = scatter._zoom.scales();
      if (!scales?.x_ || !scales?.y_) return;

      const w = window.innerWidth;
      const h = window.innerHeight;

      const t = scatter._zoom.transform;
      const k = t?.k || 1;
      setZoomLevel(k);

      const visibleLevel = k < 1.5 ? 0 : k < 4 ? 1 : 2;

      const positions = allLabels
        .filter(lb => lb.level <= visibleLevel)
        .map(lb => {
          const sx = scales.x_(lb.x);
          const sy = scales.y_(lb.y);
          return {
            ...lb,
            screenX: sx,
            screenY: sy,
            visible: sx > -80 && sx < w + 80 && sy > -30 && sy < h + 30,
          };
        })
        .filter(lb => lb.visible);

      const accepted = [];
      const MIN_DIST = 60;
      for (const lb of positions) {
        const tooClose = accepted.some(a =>
          Math.abs(a.screenX - lb.screenX) < MIN_DIST &&
          Math.abs(a.screenY - lb.screenY) < MIN_DIST * 0.6
        );
        if (!tooClose) accepted.push(lb);
      }

      setLabelPositions(accepted);
    } catch (e) { /* scales not ready */ }
  }, [allLabels]);

  // ── Initialize DeepScatter ──
  const initScatterplot = useCallback(() => {
    if (scatterRef.current) {
      try { scatterRef.current.destroy(); } catch (e) { /* ok */ }
    }

    const el = document.getElementById('deepscatter');
    if (!el) return;

    const w = el.clientWidth || window.innerWidth;
    const h = el.clientHeight || window.innerHeight;
    const scatterplot = new Scatterplot('#deepscatter', w, h);
    scatterRef.current = scatterplot;

    scatterplot.tooltip_html = (point) => {
      setSelectedPoint({
        title: point.title || 'Untitled',
        year: point.year || '',
        faculty: point.faculty || '',
        abstract: point.abstract || '',
        url: point.url || '',
        cluster: point.cluster ?? 0,
        source: point.source || '',
      });
      return '';
    };

    scatterplot.on_zoom = () => updateLabelPositions();

    const n = clusterMeta.length || 1;
    const c = colors.length ? colors : generateColors(n);

    scatterplot.plotAPI({
      source_url: 'http://localhost:8000/data/tiles',
      max_points: 1000000,
      point_size: 3,
      alpha: 55,
      zoom_balance: 0.35,
      background_color: '#0a0a0f',
      duration: 1000,
      encoding: {
        x: { field: 'x', transform: 'literal' },
        y: { field: 'y', transform: 'literal' },
        color: {
          field: 'cluster',
          range: c,
          domain: [0, Math.max(n - 1, 1)],
        },
      },
    });

    const poll = setInterval(() => {
      try {
        if (scatterplot._zoom?.scales()) {
          updateLabelPositions();
          clearInterval(poll);
        }
      } catch (e) { /* wait */ }
    }, 300);

    return () => clearInterval(poll);
  }, [clusterMeta, colors, updateLabelPositions]);

  // Re-init DeepScatter when view changes to atlas or compare
  useEffect(() => {
    if (!clusterMeta.length) return;
    if (activeView === 'som') return; // SOM doesn't need DeepScatter

    // Small delay to let DOM render the #deepscatter div
    const timer = setTimeout(() => {
      const cleanup = initScatterplot();
      return () => { if (cleanup) cleanup(); };
    }, 100);

    let resizeTimer;
    const onResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(initScatterplot, 300);
    };
    window.addEventListener('resize', onResize);

    return () => {
      clearTimeout(timer);
      clearTimeout(resizeTimer);
      window.removeEventListener('resize', onResize);
      if (scatterRef.current?.destroy) {
        try { scatterRef.current.destroy(); } catch (e) {}
      }
    };
  }, [initScatterplot, clusterMeta, activeView]);

  // ── Click cluster to highlight ──
  const handleClusterClick = useCallback((clusterId) => {
    const scatter = scatterRef.current;
    if (!scatter) return;
    const c = colors.length ? colors : generateColors(clusterMeta.length);

    if (activeCluster === clusterId) {
      setActiveCluster(null);
      scatter.plotAPI({
        encoding: {
          foreground: null,
          color: { field: 'cluster', range: c, domain: [0, Math.max(clusterMeta.length - 1, 1)] },
        },
        duration: 500,
      });
    } else {
      setActiveCluster(clusterId);
      scatter.plotAPI({
        encoding: {
          foreground: { field: 'cluster', op: 'eq', a: clusterId },
        },
        duration: 500,
      });
    }
  }, [activeCluster, colors, clusterMeta]);

  const totalDocs = clusterMeta.reduce((s, c) => s + c.count, 0);

  /* ── View switching ── */
  const handleViewChange = (view) => {
    // Destroy scatter when leaving atlas/compare
    if (view === 'som' && scatterRef.current?.destroy) {
      try { scatterRef.current.destroy(); } catch (e) {}
      scatterRef.current = null;
    }
    setActiveView(view);
  };

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', backgroundColor: '#0a0a0f' }}>

      {/* ── Header ── */}
      <header className="atlas-header">
        <div className="logo">◆ Semantic Research Atlas</div>
        <div className="header-nav">
          <button
            className={`nav-btn ${activeView === 'atlas' ? 'active' : ''}`}
            onClick={() => handleViewChange('atlas')}
          >
            🗺 Atlas
          </button>
          <button
            className={`nav-btn ${activeView === 'som' ? 'active' : ''}`}
            onClick={() => handleViewChange('som')}
          >
            🧠 SOM
          </button>
          <button
            className={`nav-btn ${activeView === 'compare' ? 'active' : ''}`}
            onClick={() => handleViewChange('compare')}
          >
            ⚡ Compare
          </button>
        </div>
        <div className="map-name">
          <span>UNAM Research Map</span> · {totalDocs.toLocaleString()} documents
        </div>
      </header>

      {/* ── Sidebar Toggle ── */}
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        style={{ left: sidebarOpen ? '352px' : '12px' }}
      >
        {sidebarOpen ? '◂' : '▸'}
      </button>

      {/* ── Sidebar ── */}
      <aside className={`atlas-sidebar ${sidebarOpen ? '' : 'collapsed'}`}>
        {activeCluster !== null && (
          <div className="active-filter">
            <span className="filter-icon">⚡</span>
            <span>Filtering: <strong>{clusterMeta.find(c => c.cluster === activeCluster)?.label}</strong></span>
            <button className="filter-clear" onClick={() => handleClusterClick(activeCluster)}>✕</button>
          </div>
        )}

        <div className="sidebar-section">
          <div className="sidebar-section-title">Data Preview</div>
          {selectedPoint ? (
            <div className="point-detail">
              <div className="detail-title" dangerouslySetInnerHTML={renderTextWithMath(selectedPoint.title)} />
              {selectedPoint.year && (
                <div className="detail-field">
                  <div className="detail-label">Year</div>
                  <div className="detail-value">{selectedPoint.year}</div>
                </div>
              )}
              {selectedPoint.faculty && (
                <div className="detail-field">
                  <div className="detail-label">Faculty / Publisher</div>
                  <div className="detail-value">{selectedPoint.faculty}</div>
                </div>
              )}
              {selectedPoint.source && (
                <div className="detail-field">
                  <div className="detail-label">Source</div>
                  <div className="detail-value source-badge">{sourceLabel(selectedPoint.source)}</div>
                </div>
              )}
              <div className="detail-field">
                <div className="detail-label">Topic</div>
                <div className="detail-value" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span className="cluster-dot" style={{ backgroundColor: colors[selectedPoint.cluster] || '#888' }} />
                  {clusterMeta.find(c => c.cluster === selectedPoint.cluster)?.label || `Cluster ${selectedPoint.cluster}`}
                </div>
              </div>
              {selectedPoint.url && (
                <div className="detail-field">
                  <div className="detail-label">Link</div>
                  <div className="detail-value">
                    <a href={selectedPoint.url} target="_blank" rel="noopener noreferrer">Open in source ↗</a>
                  </div>
                </div>
              )}
              {selectedPoint.abstract && (
                <div className="detail-field">
                  <div className="detail-label">Abstract</div>
                  <div
                    className="detail-abstract"
                    dangerouslySetInnerHTML={renderTextWithMath(
                      selectedPoint.abstract.length > 500
                        ? selectedPoint.abstract.slice(0, 500) + '…'
                        : selectedPoint.abstract
                    )}
                  />
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state">
              <div className="icon">🔍</div>
              {activeView === 'som'
                ? 'Hover over a SOM cell to see details.'
                : 'Hover over a point to see details.'}
              <br/>Click a topic label to filter.
            </div>
          )}
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-title">Topics</div>
          <div className="cluster-legend">
            {clusterMeta.map((cl) => (
              <div
                className={`cluster-item ${activeCluster === cl.cluster ? 'active' : ''}`}
                key={cl.cluster}
                onClick={() => handleClusterClick(cl.cluster)}
              >
                <span className="cluster-dot" style={{ backgroundColor: colors[cl.cluster] || '#888' }} />
                <span className="cluster-label">{cl.label}</span>
                <span className="cluster-count">{cl.count.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* ══════════════════════════════════════════════════ */}
      {/* ── VIEW: Atlas (DeepScatter) ── */}
      {/* ══════════════════════════════════════════════════ */}
      {activeView === 'atlas' && (
        <>
          <div id="deepscatter" />

          {/* Floating Topic Labels */}
          <div className="topic-labels-layer">
            {labelPositions.map((lb, i) => (
              <div
                key={`${lb.cluster}-${lb.level}-${i}`}
                className={[
                  'topic-label',
                  `topic-label--${lb.size}`,
                  activeCluster === lb.cluster ? 'active' : '',
                  activeCluster !== null && activeCluster !== lb.cluster ? 'dimmed' : '',
                ].join(' ')}
                style={{
                  left: lb.screenX,
                  top: lb.screenY,
                  '--label-color': colors[lb.cluster] || '#fff',
                }}
                onClick={(e) => { e.stopPropagation(); handleClusterClick(lb.cluster); }}
              >
                {lb.text}
              </div>
            ))}
          </div>
        </>
      )}

      {/* ══════════════════════════════════════════════════ */}
      {/* ── VIEW: SOM (U-Matrix Heatmap) ── */}
      {/* ══════════════════════════════════════════════════ */}
      {activeView === 'som' && (
        <div className="som-view">
          <div className="som-view-header">
            <h2>Self-Organizing Map · U-Matrix</h2>
            <p>Topographic visualization of the embedding space. Colors represent boundary distances between neurons — bright regions are cluster boundaries, dark regions are homogeneous areas.</p>
          </div>
          {somData ? (
            <SomHeatmap
              somData={somData}
              colors={colors}
              clusterMeta={clusterMeta}
              onPointHover={setSelectedPoint}
              width={Math.min(window.innerWidth - (sidebarOpen ? 400 : 60), 900)}
              height={Math.min(window.innerHeight - 160, 600)}
            />
          ) : (
            <div className="som-loading">
              <div className="empty-state">
                <div className="icon">🧠</div>
                SOM data not available.<br/>
                Run: <code>python scripts/04_som.py --config config/default.yaml</code>
              </div>
            </div>
          )}
          <div className="som-colorbar">
            <span>Low distance</span>
            <div className="som-gradient" />
            <span>High distance</span>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════ */}
      {/* ── VIEW: Compare (Atlas + SOM side by side) ── */}
      {/* ══════════════════════════════════════════════════ */}
      {activeView === 'compare' && (
        <div className="compare-view">
          <div className="compare-panel">
            <div className="compare-panel-header">UMAP · DeepScatter</div>
            <div id="deepscatter" />
            <div className="topic-labels-layer">
              {labelPositions.map((lb, i) => (
                <div
                  key={`cmp-${lb.cluster}-${lb.level}-${i}`}
                  className={[
                    'topic-label topic-label--small',
                    activeCluster === lb.cluster ? 'active' : '',
                    activeCluster !== null && activeCluster !== lb.cluster ? 'dimmed' : '',
                  ].join(' ')}
                  style={{
                    left: lb.screenX,
                    top: lb.screenY,
                    '--label-color': colors[lb.cluster] || '#fff',
                  }}
                  onClick={(e) => { e.stopPropagation(); handleClusterClick(lb.cluster); }}
                >
                  {lb.text}
                </div>
              ))}
            </div>
          </div>
          <div className="compare-divider" />
          <div className="compare-panel">
            <div className="compare-panel-header">SOM · U-Matrix</div>
            {somData ? (
              <SomHeatmap
                somData={somData}
                colors={colors}
                clusterMeta={clusterMeta}
                onPointHover={setSelectedPoint}
                width={Math.floor((window.innerWidth - (sidebarOpen ? 380 : 40)) / 2) - 20}
                height={window.innerHeight - 120}
              />
            ) : (
              <div className="som-loading">
                <div className="empty-state">
                  <div className="icon">🧠</div>
                  SOM data not available.
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Bottom Stats ── */}
      <div className="atlas-stats">
        <div className="stat-chip">
          <span className="stat-value">{totalDocs.toLocaleString()}</span> documents
        </div>
        <div className="stat-chip">
          <span className="stat-value">{clusterMeta.length}</span> topics
        </div>
        <div className="stat-chip">
          <span className="stat-value">3</span> sources
        </div>
      </div>
    </div>
  );
}

export default App;
