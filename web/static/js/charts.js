// Renders each <div data-chart="id"> from the <script type="application/json" id="id"> that
// Django's json_script wrote, so chart data is never inlined as JavaScript. The look (font,
// colours, gridlines) comes from the page's design tokens, and charts redraw when the theme
// changes (static/js/theme.js). A colour written as "token(--name)" in a specification
// (audit/charts.py) becomes that token's value in the current theme.
//
// States: each chart shows a skeleton until drawn, and a short message if it cannot be drawn
// (Vega failed to load, or a bad specification), so a card is never silently blank. Every
// chart is an image to assistive technology, labelled by the card's heading; the figures
// themselves are in the page's text (tiles, legends, captions).
(function () {
  const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function theme() {
    const ink = token("--ink"), muted = token("--muted"), line = token("--line");
    const palette = [token("--accent"), token("--sky"), token("--sun"), token("--teal"), token("--violet"), token("--red"), muted];
    const axis = {domainColor: line, gridColor: line, gridDash: [3, 4], tickColor: line, labelColor: muted,
                  titleColor: muted, labelFont: "Inter Variable", titleFont: "Inter Variable", labelFontSize: 12,
                  titleFontSize: 12, titleFontWeight: 500, labelLimit: 220, labelPadding: 6};
    return {
      background: null, font: "Inter Variable",
      view: {stroke: null},
      axis, axisX: {grid: false}, axisY: {domain: false, ticks: false},
      legend: {labelColor: muted, titleColor: muted, labelFont: "Inter Variable", titleFont: "Inter Variable",
               symbolType: "circle", labelFontSize: 12, titleFontSize: 12, titleFontWeight: 500},
      title: {color: ink, font: "Inter Variable", fontWeight: 600, anchor: "start"},
      range: {category: palette},
      mark: {color: palette[0]},
      bar: {color: palette[0], cornerRadiusEnd: 5},
      arc: {stroke: token("--surface"), strokeWidth: 2},
      area: {color: palette[0], opacity: 0.25, line: true},
      line: {color: palette[0], strokeWidth: 2.5},
      point: {color: palette[0], filled: true},
    };
  }

  // Replace "token(--name)" strings anywhere in the specification with the token's value.
  function resolve(value) {
    if (typeof value === "string") {
      const m = /^token\((--[\w-]+)\)$/.exec(value);
      return m ? token(m[1]) || value : value;
    }
    if (Array.isArray(value)) return value.map(resolve);
    if (value && typeof value === "object") {
      const out = {};
      for (const k in value) out[k] = resolve(value[k]);
      return out;
    }
    return value;
  }

  function label(el) {
    const heading = el.closest("section, .card")?.querySelector("h2, h3");
    el.setAttribute("role", "img");
    if (heading && !el.hasAttribute("aria-label")) el.setAttribute("aria-label", heading.textContent.trim() + " chart");
  }

  function fail(el) {
    el.classList.remove("chart-loading");
    el.innerHTML = '<p class="chart-error">This chart could not be drawn. The figures it shows are listed on this page.</p>';
  }

  function render(root = document) {
    root.querySelectorAll("[data-chart]").forEach((el) => {
      const source = document.getElementById(el.dataset.chart);
      if (!source) return;
      label(el);
      if (typeof vegaEmbed !== "function") return fail(el);
      let spec;
      try { spec = resolve(JSON.parse(source.textContent)); } catch (e) { return fail(el); }
      vegaEmbed(el, spec, {actions: false, renderer: "svg", config: theme()})
        .then(() => el.classList.remove("chart-loading"))
        .catch(() => fail(el));
    });
  }

  document.querySelectorAll("[data-chart]").forEach((el) => el.classList.add("chart-loading"));
  render();
  window.addEventListener("themechange", () => render());
  // A chart inside a closed <details> has no width yet; draw it when the section opens.
  document.querySelectorAll("details").forEach((d) => d.addEventListener("toggle", () => d.open && render(d)));
})();
