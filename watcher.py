from watchgod import watch
import os

for changes in watch('../audioGeneration'):
    os.system('python3 terminal_call.py')
    
