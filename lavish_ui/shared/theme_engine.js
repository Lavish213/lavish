/ shared/_theme_engine.js
// Handles theme init + remembering user preference

(function () {
  const THEME_KEY = "lavish-theme";
  const root = document.documentElement;

  function applyTheme(theme) {
    if (!root) return;
    if (theme === "light") {
      root.classList.add("lavish-theme-light");
      root.classList.remove("lavish-theme-dark");
    } else {
      root.classList.add("lavish-theme-dark");
      root.classList.remove("lavish-theme-light");
    }
  }

  function detectInitialTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;

    const prefersDark = window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches;
    return prefersDark ? "dark" : "dark"; // default to dark for Lavish
  }

  const initial = detectInitialTheme();
  applyTheme(initial);

  window.LavishTheme = {
    get: () => (root.classList.contains("lavish-theme-light") ? "light" : "dark"),
    set: (theme) => {
      applyTheme(theme);
      localStorage.setItem(THEME_KEY, theme);
    },
    toggle: () => {
      const next =
        root.classList.contains("lavish-theme-light") ? "dark" : "light";
      applyTheme(next);
      localStorage.setItem(THEME_KEY, next);
    },
  };
})();