FROM node:20-slim AS build

WORKDIR /app

COPY web/package*.json ./
RUN npm ci

COPY web/ ./

# ARG VITE_API_BASE_URL="http://localhost:8000"
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

RUN npm run build

FROM nginx:1.27-alpine

ENV NODE_ENV=production

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
