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
    tot = 100000
    while ct < tot:
        
        print(ct / tot)
        ct += 1
finally:
    os.unlink(pidfile)


def clean_up():
    os.unlink(pidfile)
    
import atexit
atexit.register(clean_up)

signal(SIGTERM, clean_up)
