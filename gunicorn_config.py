bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
timeout = 60
keepalive = 5
errorlog = "gunicorn.error.log"
accesslog = "gunicorn.access.log"
capture_output = True
loglevel = "info"
