const screens = document.querySelectorAll(".screen");
const nextButtons = document.querySelectorAll(".next-btn");

function showScreen(targetScreen) {
  screens.forEach((screen) => {
    screen.classList.toggle("active", screen.dataset.screen === targetScreen);
  });
}

nextButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const next = button.dataset.next;
    if (next) {
      showScreen(next);
    }
  });
});
