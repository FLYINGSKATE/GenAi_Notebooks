import React, { useState, useRef, useEffect } from 'react';
import { 
  Container, 
  TextField, 
  Button, 
  Paper, 
  Typography, 
  Box, 
  AppBar, 
  Toolbar, 
  IconButton, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  Divider, 
  Slider, 
  Switch, 
  FormControlLabel,
  CircularProgress,
  Snackbar,
  Alert,
  Avatar,
  Card,
  CardContent
} from '@mui/material';
import {
  Menu as MenuIcon,
  Send as SendIcon,
  Settings as SettingsIcon,
  Person as PersonIcon,
  History as HistoryIcon,
  Help as HelpIcon,
  Code as ApiIcon,
  GitHub as GitHubIcon,
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  SmartToy as BotIcon,
  Person as UserIcon
} from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './App.css';

// Theme configuration
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 500,
    },
  },
});

// Main App component
function App() {
  // State management
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([
    { 
      role: 'assistant', 
      content: 'Hello! I\'m your AI assistant. How can I help you today?',
      timestamp: new Date().toISOString()
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [settings, setSettings] = useState({
    apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',
    apiKey: process.env.REACT_APP_API_KEY || '',
    maxLength: 150,
    temperature: 0.7,
    topP: 0.9,
    topK: 50,
    darkMode: false,
    enableMarkdown: true,
  });
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info',
  });

  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle sending messages
  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    // Update UI with user message
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(
        `${settings.apiUrl}/generate`,
        {
          prompt: input,
          max_length: settings.maxLength,
          temperature: settings.temperature,
          top_p: settings.topP,
          top_k: settings.topK,
        },
        {
          headers: {
            'Authorization': `Bearer ${settings.apiKey}`,
            'Content-Type': 'application/json',
          },
        }
      );

      const assistantMessage = {
        role: 'assistant',
        content: response.data.generated_text,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error calling API:', error);
      setSnackbar({
        open: true,
        message: `Error: ${error.response?.data?.detail || error.message}`,
        severity: 'error',
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle key press (Enter to send)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Toggle dark/light mode
  const toggleDarkMode = () => {
    setSettings(prev => ({
      ...prev,
      darkMode: !prev.darkMode
    }));
  };

  // Render message content with markdown support
  const renderMessageContent = (content) => {
    if (settings.enableMarkdown) {
      return <ReactMarkdown>{content}</ReactMarkdown>;
    }
    return <Typography variant="body1" style={{ whiteSpace: 'pre-wrap' }}>{content}</Typography>;
  };

  // Render a single message
  const renderMessage = (message, index) => (
    <Box
      key={index}
      sx={{
        display: 'flex',
        justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          maxWidth: '80%',
        }}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            mb: 0.5,
            justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
          }}
        >
          <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: message.role === 'user' ? 'primary.main' : 'secondary.main' }}>
            {message.role === 'user' ? <UserIcon /> : <BotIcon />}
          </Avatar>
          <Typography variant="caption" color="text.secondary">
            {message.role === 'user' ? 'You' : 'Assistant'}
          </Typography>
        </Box>
        <Paper
          elevation={2}
          sx={{
            p: 2,
            borderRadius: 2,
            backgroundColor: message.role === 'user' ? 'primary.light' : 'background.paper',
            color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
          }}
        >
          {renderMessageContent(message.content)}
          <Typography variant="caption" display="block" sx={{ mt: 1, opacity: 0.7, textAlign: 'right' }}>
            {new Date(message.timestamp).toLocaleTimeString()}
          </Typography>
        </Paper>
      </Box>
    </Box>
  );

  // Settings drawer
  const renderDrawer = () => (
    <Drawer
      anchor="right"
      open={drawerOpen}
      onClose={() => setDrawerOpen(false)}
    >
      <Box sx={{ width: 350, p: 2 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Settings</Typography>
        
        <TextField
          fullWidth
          label="API URL"
          value={settings.apiUrl}
          onChange={(e) => setSettings({...settings, apiUrl: e.target.value})}
          margin="normal"
          size="small"
        />
        
        <TextField
          fullWidth
          label="API Key"
          type="password"
          value={settings.apiKey}
          onChange={(e) => setSettings({...settings, apiKey: e.target.value})}
          margin="normal"
          size="small"
        />
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="subtitle2" gutterBottom>Generation Parameters</Typography>
        
        <Box sx={{ mb: 2 }}>
          <Typography gutterBottom>Max Length: {settings.maxLength}</Typography>
          <Slider
            value={settings.maxLength}
            onChange={(_, value) => setSettings({...settings, maxLength: value})}
            min={1}
            max={1000}
            step={1}
            valueLabelDisplay="auto"
          />
        </Box>
        
        <Box sx={{ mb: 2 }}>
          <Typography gutterBottom>Temperature: {settings.temperature.toFixed(1)}</Typography>
          <Slider
            value={settings.temperature}
            onChange={(_, value) => setSettings({...settings, temperature: value})}
            min={0}
            max={2}
            step={0.1}
            valueLabelDisplay="auto"
          />
        </Box>
        
        <FormControlLabel
          control={
            <Switch
              checked={settings.enableMarkdown}
              onChange={(e) => setSettings({...settings, enableMarkdown: e.target.checked})}
            />
          }
          label="Enable Markdown"
        />
        
        <FormControlLabel
          control={
            <Switch
              checked={settings.darkMode}
              onChange={toggleDarkMode}
              icon={<LightModeIcon />}
              checkedIcon={<DarkModeIcon />}
            />
          }
          label={settings.darkMode ? 'Dark Mode' : 'Light Mode'}
        />
        
        <Divider sx={{ my: 2 }} />
        
        <Button
          variant="outlined"
          fullWidth
          startIcon={<GitHubIcon />}
          href="https://github.com/yourusername/llm-chat-ui"
          target="_blank"
          rel="noopener noreferrer"
          sx={{ mb: 1 }}
        >
          View on GitHub
        </Button>
      </Box>
    </Drawer>
  );

  return (
    <ThemeProvider theme={theme}>
      <Box 
        sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          minHeight: '100vh',
          backgroundColor: settings.darkMode ? '#121212' : '#f5f5f5',
          color: settings.darkMode ? '#fff' : 'inherit',
        }}
      >
        <AppBar position="static" color="primary">
          <Toolbar>
            <IconButton
              size="large"
              edge="start"
              color="inherit"
              aria-label="menu"
              sx={{ mr: 2 }}
              onClick={() => setDrawerOpen(true)}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              LLM Chat Interface
            </Typography>
            <IconButton
              color="inherit"
              onClick={toggleDarkMode}
            >
              {settings.darkMode ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        <Container 
          maxWidth="md" 
          sx={{ 
            flexGrow: 1, 
            py: 3, 
            display: 'flex', 
            flexDirection: 'column',
            height: 'calc(100vh - 128px)',
          }}
        >
          <Paper 
            elevation={3} 
            sx={{ 
              flexGrow: 1, 
              p: 2, 
              mb: 2, 
              overflowY: 'auto',
              backgroundColor: settings.darkMode ? '#1e1e1e' : '#fff',
            }}
          >
            {messages.map((message, index) => renderMessage(message, index))}
            {isLoading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                <CircularProgress size={24} />
              </Box>
            )}
            <div ref={messagesEndRef} />
          </Paper>

          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              multiline
              maxRows={4}
              disabled={isLoading}
              sx={{
                '& .MuiOutlinedInput-root': {
                  backgroundColor: settings.darkMode ? '#333' : '#fff',
                },
              }}
            />
            <Button
              variant="contained"
              color="primary"
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              sx={{ minWidth: '100px' }}
              startIcon={<SendIcon />}
            >
              {isLoading ? 'Sending...' : 'Send'}
            </Button>
          </Box>
        </Container>

        {renderDrawer()}
        
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar({...snackbar, open: false})}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert 
            onClose={() => setSnackbar({...snackbar, open: false})} 
            severity={snackbar.severity}
            sx={{ width: '100%' }}
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;
