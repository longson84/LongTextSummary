// const sidebar = document.createElement("div");
// sidebar.id = "my-sidebar";
// sidebar.innerHTML = `
//     <button id="summarize">Summarize The Page</button>
//     <div id="summary">The summary will appear here</div>
// `;
// document.body.appendChild(sidebar);
//
// // Rest of your script here (event listener, fetch call, etc.)
// // Inside your content.js, after creating the sidebar
//
// const loader = document.createElement("div");
// loader.id = "loading-indicator";
// loader.innerText = "Loading...";
// loader.style.display = "none"; // Hide initially
// document.body.appendChild(loader);
//
// document.getElementById("summarize").addEventListener("click", function () {
//   // Show the loader
//   document.getElementById("loading-indicator").style.display = "block";
//
//   chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
//     // ... your existing code ...
//
//     fetch("http://127.0.0.1:8000/api/summarize/", {
//       // ... your existing fetch configuration ...
//     })
//       .then((res) => res.json())
//       .then((data) => {
//         // Hide the loader
//         document.getElementById("loading-indicator").style.display = "none";
//
//         // ... rest of your success logic ...
//       })
//       .catch((err) => {
//         // Hide the loader
//         document.getElementById("loading-indicator").style.display = "none";
//
//         console.log("Error: ", err);
//       });
//   });
// });
