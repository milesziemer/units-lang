FROM node:16.13.0-alpine

WORKDIR /usr

COPY package.json ./
COPY .babelrc ./
COPY webpack.server.config.js ./
COPY webpack.prod.config.js ./
COPY pkg ./pkg
COPY src ./src

RUN yarn
RUN yarn buildProd


FROM node:17.3.0-alpine

WORKDIR /usr

COPY package.json ./
COPY --from=0 /usr/pkg ./pkg
RUN yarn --production

COPY --from=0 /usr/dist ./dist
RUN yarn global add pm2

ENV PORT=80
EXPOSE 80
CMD ["pm2-runtime", "./dist/server.js"]