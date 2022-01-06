import os, sys
from signal import signal, SIGTERM

pid = str(os.getpid())
pidfile = 'mydaemon.pid'
if os.path.isfile(pidfile):
    print('%s already exists, exiting' % pidfile)
    sys.exit()
open(pidfile, 'w').write(pid)
try:
    ct = 0
    while True:
        print(ct)
finally:
    os.unlink(pidfile)


def clean_up():
    os.unlink(pidfile)
    
import atexit
atexit.register(clean_up)

signal(SIGTERM, clean_up)
