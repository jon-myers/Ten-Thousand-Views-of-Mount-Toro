import subprocess

cmd = ['pgrep -f .*inf_ct.py']
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
stderr=subprocess.PIPE)
my_pid, err = process.communicate()

if len(my_pid.splitlines()) >0:
   print("Running")
   exit()
else:
  print("Not Running")
