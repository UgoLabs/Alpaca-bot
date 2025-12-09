import os
import subprocess

def check_status():
    print("🐳 Docker Bot Status Check\n")
    
    # Check running containers
    try:
        result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print("Error checking docker-compose ps")

    # Check logs
    print("\n📜 Recent Logs (Last 10 lines):")
    try:
        result = subprocess.run(['docker-compose', 'logs', '--tail=5'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print("Error checking logs")

if __name__ == "__main__":
    check_status()
