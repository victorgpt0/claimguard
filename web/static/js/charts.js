// Renders each <div data-chart="id"> from the <script type="application/json" id="id"> that
// Django's json_script wrote, so chart data is never inlined as JavaScript. The look (font,
// colours, gridlines) comes from the page's design tokens, and charts redraw when the theme
// changes (static/js/theme.js).
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

  function render() {
    document.querySelectorAll("[data-chart]").forEach((el) => {
      const source = document.getElementById(el.dataset.chart);
      if (source) vegaEmbed(el, JSON.parse(source.textContent), {actions: false, renderer: "svg", config: theme()});
    });
  }

  render();
  window.addEventListener("themechange", render);
})();
