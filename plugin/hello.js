const stopButton = document.getElementById("stop-button");
const submitButton = document.getElementById("summarize-button");
const loading = document.getElementById("loading-indicator");
const summarizing = document.getElementById("summary");

function showLoadingIndicator() {
  loading.style.display = "block";
  summarizing.style.display = "block";
  summarizing.innerText = "Summarizing, please wait";
  submitButton.disabled = true;
  stopButton.style.display = "block";
}

function hideLoadingIndicator() {
  loading.style.display = "none";
  submitButton.disabled = false;
  // summarizing.style.display = "none";
  stopButton.style.display = "none";
}

let processController;

submitButton.addEventListener("click", function () {
  processController = new AbortController();

  const { signal } = processController;

  showLoadingIndicator();

  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    // Step 1: Take the URL of the current tab
    const currentTabUrl = tabs[0].url;

    // Step 2: Send this URL to the API endpoint of the Django server

    fetch("http://127.0.0.1:8000/api/summarize/", {
      method: "POST",
      body: JSON.stringify({ url: currentTabUrl }),
      headers: { "Content-Type": "application/json" },
      signal,
    })
      .then((res) => res.json())
      .then((data) => {
        // Step 3: receive the result and display it
        summarizing.style.display = "block";
        summarizing.innerText = data.result;
      })
      .catch((err) => console.log("Error: ", err))
      .finally(() => {
        hideLoadingIndicator();
      });
  });
});

stopButton.addEventListener("click", function () {
  if (processController) {
    processController.abort();
    hideLoadingIndicator();
  }
});
