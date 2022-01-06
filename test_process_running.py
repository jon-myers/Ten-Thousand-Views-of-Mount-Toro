import psutil

def check_if_process_running(process_name):
    for proc in psutil.process_iter():
        try:
            if process_name.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return False;
        
if check_if_process_running('terminal_call.py'):
    print('currently running')
else:
    print('not currently running')
