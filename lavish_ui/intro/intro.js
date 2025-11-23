// lavish_ui/intro/intro.js

// ---------------- Matrix background ----------------

(function initMatrix() {
  const canvas = document.getElementById("matrix-canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const letters = "01⌁ΛVⅠ$H¥";
  let width, height, columns, drops;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    columns = Math.floor(width / 16);
    drops = new Array(columns).fill(1);
  }

  resize();
  window.addEventListener("resize", resize);

  function draw() {
    ctx.fillStyle = "rgba(0, 0, 0, 0.16)";
    ctx.fillRect(0, 0, width, height);

    ctx.font = "14px monospace";

    for (let i = 0; i < drops.length; i++) {
      const text = letters.charAt(Math.floor(Math.random() * letters.length));
      const x = i * 16;
      const y = drops[i] * 18;

      const gradient = ctx.createLinearGradient(x, y - 20, x, y + 10);
      gradient.addColorStop(0, "rgba(74, 222, 128, 0.12)");
      gradient.addColorStop(1, "rgba(34, 197, 94, 0.85)");
      ctx.fillStyle = gradient;
      ctx.fillText(text, x, y);

      if (y > height && Math.random() > 0.975) {
        drops[i] = 0;
      }
      drops[i]++;
    }

    requestAnimationFrame(draw);
  }

  requestAnimationFrame(draw);
})();

// ---------------- Boot progress + transition ----------------

(function initIntro() {
  const progressBar = document.getElementById("intro-progress-fill");
  const progressLabel = document.getElementById("intro-progress-label");
  const enterBtn = document.getElementById("enter-console-btn");
  const introRoot = document.getElementById("intro-root");

  let progress = 0;
  let hasExited = false;

  function setProgress(value) {
    progress = Math.min(100, Math.max(0, value));
    if (progressBar) {
      progressBar.style.width = `${progress}%`;
      progressBar.style.backgroundPosition = `${100 - progress}% 50%`;
    }
    if (progressLabel) {
      progressLabel.textContent = `Booting Lavish Core… ${progress}%`;
    }
  }

  // Smooth fake boot up to 100%
  function startBoot() {
    const interval = setInterval(() => {
      if (progress >= 100) {
        clearInterval(interval);
        if (progressLabel) {
          progressLabel.textContent = "Lavish Core ready.";
        }
        return;
      }
      const step = progress < 70 ? 6 : progress < 92 ? 3 : 1;
      setProgress(progress + step);
    }, 180);
  }

  startBoot();

  // Navigation with quick fade
  function goToDashboard(reason) {
    if (hasExited) return;
    hasExited = true;

    if (introRoot) {
      introRoot.classList.add("intro-exit");
    }

    // small delay for hard-ish cut
    setTimeout(() => {
      window.location.href = "/dashboard";
    }, 260);
  }

  if (enterBtn) {
    enterBtn.addEventListener("click", () => {
      goToDashboard("button");
    });
  }

  // Failsafe auto-redirect (in case user doesn't click)
  setTimeout(() => {
    goToDashboard("auto");
  }, 9000);
})();
