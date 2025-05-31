"""
Ghost - Minimal System Cleaner for Windows
Copyright (c) 2025 Shizuku Tanaka
MIT License
"""

import os
import sys
import time
import shutil
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

if sys.platform != "win32":
    print("Ghost is for Windows only.")
    sys.exit(1)

try:
    import psutil
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    import psutil
    import pystray
    from PIL import Image, ImageDraw


class Ghost:
    """Minimal system cleaner."""
    
    def __init__(self):
        self.running = False
        self.total_cleaned_mb = 0
        self.last_cleanup = None
        
        self.temp_dirs = [
            Path(tempfile.gettempdir()),
            Path(os.environ.get('TEMP', '')),
            Path.home() / 'AppData' / 'Local' / 'Temp',
            Path('C:/Windows/Temp'),
        ]
        
        self.browser_caches = {
            'Chrome': Path.home() / 'AppData/Local/Google/Chrome/User Data/Default/Cache',
            'Edge': Path.home() / 'AppData/Local/Microsoft/Edge/User Data/Default/Cache',
            'Firefox': Path.home() / 'AppData/Local/Mozilla/Firefox/Profiles',
        }
    
    def create_icon(self) -> Image.Image:
        """Create tray icon."""
        image = Image.new('RGB', (64, 64), 'black')
        draw = ImageDraw.Draw(image)
        # Ghost shape
        draw.ellipse([16, 16, 48, 48], fill='white')
        draw.ellipse([22, 26, 28, 32], fill='black')  # Left eye
        draw.ellipse([36, 26, 42, 32], fill='black')  # Right eye
        return image
    
    def get_system_info(self) -> Dict[str, float]:
        """Get system metrics."""
        return {
            'cpu': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('C:').percent
        }
    
    def clean_temp_files(self) -> Tuple[int, float]:
        """Clean temporary files."""
        files_deleted = 0
        space_freed = 0
        cutoff = time.time() - 86400  # 24 hours
        
        for temp_dir in self.temp_dirs:
            if not temp_dir.exists():
                continue
                
            try:
                for file in temp_dir.rglob('*'):
                    if file.is_file():
                        try:
                            if file.stat().st_mtime > cutoff:
                                continue
                            size = file.stat().st_size
                            file.unlink()
                            files_deleted += 1
                            space_freed += size
                        except:
                            pass
            except:
                pass
        
        return files_deleted, space_freed / 1048576  # Convert to MB
    
    def clean_browser_cache(self) -> float:
        """Clean browser cache."""
        space_freed = 0
        
        # Check for running browsers
        for proc in psutil.process_iter(['name']):
            name = proc.info.get('name', '').lower()
            if name in ['chrome.exe', 'msedge.exe', 'firefox.exe']:
                return 0
        
        for browser, path in self.browser_caches.items():
            if not path.exists():
                continue
                
            try:
                if browser == 'Firefox':
                    for profile in path.iterdir():
                        cache = profile / 'cache2'
                        if cache.exists():
                            shutil.rmtree(cache, ignore_errors=True)
                else:
                    for file in path.rglob('*'):
                        if file.is_file():
                            try:
                                size = file.stat().st_size
                                file.unlink()
                                space_freed += size
                            except:
                                pass
            except:
                pass
        
        return space_freed / 1048576
    
    def clean_all(self) -> Dict[str, float]:
        """Run full cleanup."""
        files, temp_mb = self.clean_temp_files()
        browser_mb = self.clean_browser_cache()
        
        total_mb = temp_mb + browser_mb
        self.total_cleaned_mb += total_mb
        self.last_cleanup = datetime.now()
        
        return {
            'files': files,
            'size_mb': total_mb
        }
    
    def show_status(self, icon, item):
        """Show status window."""
        info = self.get_system_info()
        msg = f"""Ghost - System Status

CPU: {info['cpu']:.1f}%
Memory: {info['memory']:.1f}%
Disk: {info['disk']:.1f}%

Total Cleaned: {self.total_cleaned_mb:.1f} MB
Last Run: {self.last_cleanup.strftime('%H:%M') if self.last_cleanup else 'Never'}"""
        
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "Ghost", 0x40)
    
    def run_cleanup(self, icon, item):
        """Run cleanup."""
        icon.notify("Cleaning...", "Ghost")
        result = self.clean_all()
        icon.notify(f"Freed {result['size_mb']:.1f} MB", "Ghost")
    
    def auto_clean(self):
        """Auto cleanup thread."""
        while self.running:
            time.sleep(3600)  # 1 hour
            if psutil.cpu_percent(1) < 30:  # System idle
                self.clean_all()
    
    def quit(self, icon, item):
        """Exit application."""
        self.running = False
        icon.stop()
    
    def run(self):
        """Run tray application."""
        self.running = True
        
        threading.Thread(target=self.auto_clean, daemon=True).start()
        
        menu = pystray.Menu(
            pystray.MenuItem("Status", self.show_status),
            pystray.MenuItem("Clean Now", self.run_cleanup),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self.quit)
        )
        
        icon = pystray.Icon("Ghost", self.create_icon(), menu=menu)
        icon.run()


def main():
    if len(sys.argv) > 1:
        ghost = Ghost()
        if sys.argv[1] == '--clean':
            print("Ghost - Running cleanup...")
            result = ghost.clean_all()
            print(f"✓ Deleted {result['files']} files")
            print(f"✓ Freed {result['size_mb']:.1f} MB")
        elif sys.argv[1] == '--status':
            info = ghost.get_system_info()
            print(f"CPU: {info['cpu']:.1f}%")
            print(f"Memory: {info['memory']:.1f}%") 
            print(f"Disk: {info['disk']:.1f}%")
        else:
            print("Usage: ghost.py [--clean|--status]")
    else:
        Ghost().run()


if __name__ == "__main__":
    main()