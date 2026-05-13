import { useEffect, useRef, useState, useCallback } from 'react';
import Scatterplot from 'deepscatter';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import './index.css';

/**
 * Generate N visually distinct, bright colors via golden-angle HSL.
 * Avoids dark/invisible colors by enforcing high saturation and lightness.
 */
function generateColors(n) {
  const golden = 137.508;
  const colors = [];
  for (let i = 0; i < n; i++) {
    const hue = (i * golden) % 360;
    const sat = 70 + (i % 3) * 8;   // 70-86%
    const lum = 60 + (i % 2) * 8;   // 60-68% — always bright
    colors.push(`hsl(${hue}, ${sat}%, ${lum}%)`);
  }
  return colors;
}

/**
 * Render text with HTML and LaTeX math formulas parsed via KaTeX
 */
function renderTextWithMath(text) {
  if (!text) return { __html: '' };
  
  let html = text;
  
  // Clean up rogue XML tags that OpenAlex sometimes includes
  html = html.replace(/&lt;\/?title&gt;/gi, '');
  html = html.replace(/<\/?title>/gi, '');
  html = html.replace(/&lt;\/?jats:[a-z]+&gt;/gi, '');
  html = html.replace(/<\/?jats:[a-z]+>/gi, '');

  try {
    // Parse LaTeX $$...$$
    html = html.replace(/\$\$(.*?)\$\$/g, (match, math) => {
      return katex.renderToString(math, { throwOnError: false, displayMode: true });
    });
    // Parse LaTeX $...$
    html = html.replace(/\$(.*?)\$/g, (match, math) => {
      // Ignore matches that look like prices e.g., $10
      if (math.match(/^\d/)) return match;
      return katex.renderToString(math, { throwOnError: false, displayMode: false });
    });
  } catch (e) {
    // Ignore KaTeX parsing errors
  }
  return { __html: html };
}

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

  // ── Convert data coords → screen coords ──
  const updateLabelPositions = useCallback(() => {
    const scatter = scatterRef.current;
    if (!scatter?._zoom || !allLabels.length) return;

    try {
      const scales = scatter._zoom.scales();
      if (!scales?.x_ || !scales?.y_) return;

      const w = window.innerWidth;
      const h = window.innerHeight;

      // Get current zoom scale factor
      const t = scatter._zoom.transform;
      const k = t?.k || 1;
      setZoomLevel(k);

      // Determine which label levels are visible based on zoom
      // level 0 = always, level 1 = medium zoom, level 2 = close zoom
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

      // Simple collision avoidance: remove labels too close to each other
      const accepted = [];
      const MIN_DIST = 60; // pixels
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

    const w = window.innerWidth;
    const h = window.innerHeight;
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

    // Real-time label tracking on zoom/pan
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

    // Poll until scales are ready for initial label placement
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

  useEffect(() => {
    if (!clusterMeta.length) return;
    const cleanup = initScatterplot();
    let timer;
    const onResize = () => {
      clearTimeout(timer);
      timer = setTimeout(initScatterplot, 300);
    };
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      clearTimeout(timer);
      if (cleanup) cleanup();
      if (scatterRef.current?.destroy) {
        try { scatterRef.current.destroy(); } catch(e) {}
      }
    };
  }, [initScatterplot, clusterMeta]);

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

  const sourceLabel = (s) => ({
    openalex: 'OpenAlex',
    unam_repository: 'Repositorio UNAM',
    scielo_mexico: 'SciELO México',
  }[s] || s);

  const totalDocs = clusterMeta.reduce((s, c) => s + c.count, 0);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', backgroundColor: '#0a0a0f' }}>
      <div id="deepscatter" />

      {/* ── Floating Topic Labels ── */}
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

      {/* ── Header ── */}
      <header className="atlas-header">
        <div className="logo">◆ Semantic Research Atlas</div>
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
              Hover over a point to see details.<br/>Click a topic label to filter.
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
