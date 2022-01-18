import "./css/generated.css";

if (typeof module.hot !== "undefined") {
  module.hot.accept();
}

import("units-lang")
  .then(({ Runtime }) => {
    const runtime = Runtime.new();
    const interpElem = document.getElementById("interpreter");
    const inputElem = document.getElementById("console");
    const history = document.getElementById("history");
    inputElem.onfocus = (e) => {
      e.preventDefault();
      interpElem.style.boxShadow = "0 -0.5px 0 #000, 0px 0.5px 0 #000";
    };
    inputElem.onblur = (e) => {
      e.preventDefault();
      interpElem.style.boxShadow = "";
    };
    interpElem.onsubmit = (e) => {
      e.preventDefault();
      const input = inputElem.value;
      const echoElem = document.createElement("li");
      echoElem.textContent = "units > ";
      echoElem.className = "lead history-list-item";
      if (input.length > 0) {
        const response = runtime.run(input);
        echoElem.textContent += `${input}`;
        const responseElem = document.createElement("div");
        responseElem.style.whiteSpace = "pre-wrap";
        responseElem.textContent = `\t > ${response}\n`;
        echoElem.appendChild(responseElem);
      }
      history.prepend(echoElem);
      inputElem.value = "";
    };
  })
  .catch((e) => console.error("Error loading App.js: ", e));
