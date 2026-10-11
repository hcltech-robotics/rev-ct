"use strict";

const copyButton = document.getElementById("copy-citation");
const copyStatus = document.getElementById("copy-status");
const bibtex = document.getElementById("bibtex");

copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim() + "\n");
    copyStatus.textContent = "Copied.";
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = "Citation selected. Press ⌘C or Ctrl+C to copy.";
  }
});
