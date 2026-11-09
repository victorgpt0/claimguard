// Light / dark / system theme switch. The choice is kept in localStorage ("system" removes
// it); base.html applies it before first paint. Charts listen for "themechange" to redraw.
document.addEventListener("alpine:init", () => {
  const media = matchMedia("(prefers-color-scheme: dark)");
  const read = () => { try { return localStorage.getItem("theme") || "system"; } catch (e) { return "system"; } };
  const apply = (choice) => {
    const theme = choice === "system" ? (media.matches ? "dark" : "light") : choice;
    if (document.documentElement.dataset.theme !== theme) {
      document.documentElement.dataset.theme = theme;
      window.dispatchEvent(new CustomEvent("themechange", {detail: theme}));
    }
  };
  Alpine.data("themeSwitch", () => ({
    choice: read(),
    open: false,
    init() { media.addEventListener("change", () => { if (this.choice === "system") apply("system"); }); },
    set(choice) {
      this.choice = choice; this.open = false;
      try { choice === "system" ? localStorage.removeItem("theme") : localStorage.setItem("theme", choice); } catch (e) {}
      apply(choice);
    },
  }));
});
