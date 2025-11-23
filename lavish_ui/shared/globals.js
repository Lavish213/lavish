// shared/_globals.js
// Shared constants & small helpers for intro + dashboard

export const LAVISH_API_BASE = "http://127.0.0.1:8000";

export const LAVISH_COLORS = {
  gold: "#f6c15c",
  green: "#2dde98",
  red: "#ff4c6a",
};

export function safeQuery(selector) {
  return document.querySelector(selector);
}

export function formatCurrency(value) {
  if (value === null || value === undefined || isNaN(value)) return "$0.00";
  return (
    "$" +
    Number(value).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })
  );
}