import "./css/generated.css";

if (typeof module.hot !== "undefined") {
  module.hot.accept();
}

import("units-lang")
  .then(({ Runtime }) => {
    const runtime = Runtime.new();
    const interpreter = document.getElementById("interpreter");
    const prompt = document.getElementById("prompt");
    const history = document.getElementById("history");
    const docsButton = document.getElementById("docsButton");
    docsButton.onclick = (e) => {
      const docs = document.getElementById("docs");
      // if (docs.style.display === "none") {
      //   document.getElementById("")
      // }
      docs.style.display = docs.style.display === "none" ? "block" : "none";
    };
    prompt.onfocus = (e) => {
      e.preventDefault();
      interpreter.style.boxShadow = "0 -0.5px 0 #000, 0px 0.5px 0 #000";
    };
    prompt.onblur = (e) => {
      e.preventDefault();
      interpreter.style.boxShadow = "";
    };
    interpreter.onsubmit = (e) => {
      e.preventDefault();
      const input = prompt.value;
      const echoElem = document.createElement("li");
      echoElem.textContent = "units > ";
      echoElem.className = "lead history-item";
      if (input.length > 0) {
        const response = runtime.run(input);
        echoElem.textContent += `${input}`;
        const responseElem = document.createElement("div");
        responseElem.style.whiteSpace = "pre-wrap";
        responseElem.textContent = `\t > ${response}\n`;
        echoElem.appendChild(responseElem);
      }
      history.prepend(echoElem);
      prompt.value = "";
    };
  })
  .catch((e) => console.error("Error loading App.js: ", e));
