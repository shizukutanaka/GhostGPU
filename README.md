# Ghost 🚀

A minimal system cleaner for Windows that focuses on what matters most. Keep your system clean and fast without the bloat.

## ✨ Features

- 🧹 **Temporary Files Cleanup** - Removes files older than 24 hours
- 🌐 **Browser Cache Cleanup** - Supports Chrome, Edge, and Firefox
- 💻 **System Tray Integration** - Always accessible when you need it
- 🤖 **Auto Cleanup** - Runs automatically when system is idle
- 🪶 **Lightweight** - Uses less than 20MB of memory
- 🛡️ **Safe** - Only cleans standard temporary directories and browser caches

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShizukuTanaka/ghost.git
   cd ghost
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Ghost**
   ```bash
   python ghost.py
   ```

## 🛠️ Usage

### System Tray Options
- **Clean Now**: Perform an immediate cleanup
- **Settings**: Configure cleanup preferences
- **View Logs**: See what was cleaned up
- **Exit**: Close the application

### Command Line Options
```
python ghost.py [--auto] [--silent] [--log-level LEVEL]
```

- `--auto`: Run in auto-cleanup mode and exit
- `--silent`: Don't show notifications
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your fork
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped improve this project
- Built with ❤️ by Shizuku Tanaka