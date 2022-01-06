import os, sys

pid = str(os.getpid())
pidfile = 'mydaemon.pid'
if os.path.isfile(pidfile):
    print('%s already exists, exiting' % pidfile)
    sys.exit()
open(pidfile, 'w').write(pid)
try:
    ct = 0
    while ct < 100000:
        ct += 1
finally:
    os.unlink(pidfile)
