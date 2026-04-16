import time
import os
import sqlite3
import signal
import subprocess

def test():
    print("Starting integration test for exit flushing...")
    
    # Ensure starting clean
    cmd = ['python3', 'main.py', '--fast', '--reset-db']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("Running main.py... Waiting 12 seconds...")
    time.sleep(12) 
    
    print("Sending SIGTERM to main.py to trigger flush_remaining_exits()...")
    p.send_signal(signal.SIGTERM)
    
    try:
        stdout, stderr = p.communicate(timeout=15)
        print("Process exited gracefully.")
    except subprocess.TimeoutExpired:
        p.kill()
        stdout, stderr = p.communicate()
        print("Process had to be killed!")
        
    print("--- STDOUT ---")
    for line in stdout.split('\n'):
        if "EXIT" in line or "flush" in line.lower() or "shutdown" in line.lower():
            print(line)
            
    # Verify DB
    conn = sqlite3.connect('faces_db/faces.db')
    c = conn.cursor()
    events = c.execute("SELECT * FROM events").fetchall()
    
    entries = [e for e in events if e[2] == 'entry']
    exits = [e for e in events if e[2] == 'exit']
    
    print("\n--- TEST RESULTS ---")
    print(f"Total entries: {len(entries)}")
    print(f"Total exits : {len(exits)}")
    print(f"All events:")
    for e in events:
        print(f"  {e}")
        
    assert len(entries) > 0, "No entries found, test didn't gather enough data."
    assert len(entries) == len(exits), f"Mismatch! Entries: {len(entries)}, Exits: {len(exits)}"
    print("TEST PASSED: all entries have a corresponding exit.")
    
if __name__ == '__main__':
    test()
