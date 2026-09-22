# Self-signed TLS certificate

The `web-https` service reverse-proxies `/v1/realtime` over WSS through
`docker/web_tls_proxy.py` and loads its certificate from the host paths
`docker/certs/server.crt` and `docker/certs/server.key`. Browsers only grant
microphone access on HTTPS for non-`localhost` origins, so a certificate pair is
required for deployment; this repository ships **no private key file**, so
generate one for your deployment host:

```bash
cd docker/certs

# Replace <HOST> with the IP or DNS name of the deployment host (the address the
# browser uses). The SAN must cover the host name the browser actually opens,
# otherwise the handshake fails.
openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
  -keyout server.key -out server.crt \
  -subj "/CN=<HOST>" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,DNS:<HOST>,IP:<HOST>"
```

Restart the `web-https` service afterwards. The browser asks you to accept this
self-signed certificate once, on the first visit to `https://<HOST>:8443/`.
