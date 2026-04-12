FROM nginx:1.27-alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our reverse-proxy config
COPY nginx.conf /etc/nginx/conf.d/dashboard.conf

# Copy the static dashboard
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80
