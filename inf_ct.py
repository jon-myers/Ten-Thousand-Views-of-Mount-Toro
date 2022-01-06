import os, sys

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
