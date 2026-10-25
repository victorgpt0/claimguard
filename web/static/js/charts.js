// Renders each <div data-chart="id"> from the <script type="application/json" id="id"> that
// Django's json_script wrote, so chart data is never inlined as JavaScript.
document.querySelectorAll("[data-chart]").forEach((el) => {
  const source = document.getElementById(el.dataset.chart);
  if (source) vegaEmbed(el, JSON.parse(source.textContent), {actions: false, renderer: "svg"});
});
