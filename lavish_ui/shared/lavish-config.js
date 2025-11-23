// Lavish global config – mock layer now, easy to swap to real API later.

window.LAVISH_CONFIG = {
  // change this to your FastAPI base later:
  apiBase: "http://127.0.0.1:8000",

  // for mock dashboard mode
  mockMode: true,

  // Market open time (local client time) – 6:30 am for PST (fake local)
  marketOpenHourLocal: 6,
  marketOpenMinuteLocal: 30,

  // XP / Level config
  xpPerWin: 25,
  xpPerSession: 10,
  xpPerTeaching: 15,

  // thresholds
  xpLevels: [
    { level: 1, name: "Novice", xp: 0 },
    { level: 2, name: "Analyst", xp: 250 },
    { level: 3, name: "Trader", xp: 600 },
    { level: 4, name: "Strategist", xp: 1200 },
    { level: 5, name: "Wall Street Elite", xp: 2500 },
  ],
};