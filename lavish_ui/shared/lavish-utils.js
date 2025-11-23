// Shared utils for intro + dashboard

(function () {
  const cfg = window.LAVISH_CONFIG || {};

  function formatCurrency(value) {
    if (value === null || value === undefined || isNaN(value)) return "$0.00";
    return (
      "$" +
      Number(value).toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })
    );
  }

  function formatPercent(v, digits = 2) {
    if (v === null || v === undefined || isNaN(v)) return "0%";
    return `${Number(v).toFixed(digits)}%`;
  }

  function getNextMarketOpenCountdown() {
    // Simplified: next local day at configured open time
    const now = new Date();
    const open = new Date();
    open.setHours(cfg.marketOpenHourLocal || 6, cfg.marketOpenMinuteLocal || 30, 0, 0);

    // if we've passed open today, move to tomorrow (skip weekends naive)
    if (open <= now) {
      open.setDate(open.getDate() + 1);
    }

    const diffMs = open - now;
    const totalSec = Math.max(0, Math.floor(diffMs / 1000));
    const hours = Math.floor(totalSec / 3600);
    const minutes = Math.floor((totalSec % 3600) / 60);
    const seconds = totalSec % 60;

    return { hours, minutes, seconds };
  }

  function computeLavishLevel(xp) {
    const levels = cfg.xpLevels || [];
    if (!levels.length) {
      return { level: 1, name: "Novice", nextLevelXp: 100, currentLevelMinXp: 0 };
    }
    let current = levels[0];
    let next = levels[levels.length - 1];

    for (let i = 0; i < levels.length; i++) {
      if (xp >= levels[i].xp) {
        current = levels[i];
        next = levels[Math.min(i + 1, levels.length - 1)];
      }
    }
    return {
      level: current.level,
      name: current.name,
      nextLevelXp: next.xp,
      currentLevelMinXp: current.xp,
    };
  }

  window.LavishUtils = {
    formatCurrency,
    formatPercent,
    getNextMarketOpenCountdown,
    computeLavishLevel,
  };
})();