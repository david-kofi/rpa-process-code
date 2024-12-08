const { app, BrowserWindow } = require('electron');

app.on('ready', () => {
  const mainWindow = new BrowserWindow({
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile('index.html');
});
