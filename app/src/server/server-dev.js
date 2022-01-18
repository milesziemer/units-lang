import path from "path";
import express from "express";
import webpack from "webpack";
import WebpackDevMiddleware from "webpack-dev-middleware";
import WebpackHotMiddleware from "webpack-hot-middleware";
import config from "../../webpack.dev.config.js";

const app = express();
const DIST_DIR = __dirname;
const HTML_FILE = path.join(DIST_DIR, "index.html");
const compiler = webpack(config);

app.use(
  WebpackDevMiddleware(compiler, {
    publicPath: config.output.publicPath,
  })
);

app.use(WebpackHotMiddleware(compiler));

app.get("*", (req, res, next) => {
  compiler.outputFileSystem.readFile(HTML_FILE, (err, result) => {
    if (err) {
      return next(err);
    }
    res.set("content-type", "text/html");
    res.send(result);
    res.end();
  });
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Listening at http://localhost:${PORT}`);
});
