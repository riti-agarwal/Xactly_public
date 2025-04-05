import React, { useState, useRef } from 'react';
import { Box, TextField, styled, Typography, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const ChatContainer = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  backgroundColor: '#2d2d2d',
});

const MessagesContainer = styled(Box)({
  flex: 1,
  overflowY: 'auto',
  padding: '20px',
  display: 'flex',
  flexDirection: 'column',
});

const InputContainer = styled(Box)({
  padding: '20px',
  borderTop: '1px solid #333',
});

const StyledInput = styled(TextField)({
  width: '100%',
  '& .MuiOutlinedInput-root': {
    color: 'white',
    backgroundColor: '#1e1e1e',
    '& fieldset': {
      borderColor: '#333',
    },
    '&:hover fieldset': {
      borderColor: '#666',
    },
    '&.Mui-focused fieldset': {
      borderColor: '#0078d4',
    },
  },
  '& .MuiInputLabel-root': {
    color: '#888',
  },
});

const ContextPreview = styled(Box)({
  display: 'flex',
  alignItems: 'center',
  padding: '8px 12px',
  backgroundColor: '#1e1e1e',
  borderRadius: '8px 8px 0 0',
  marginBottom: '4px',
});

const ContextImage = styled('img')({
  width: '40px',
  height: '40px',
  objectFit: 'cover',
  borderRadius: '4px',
  marginRight: '8px',
});

const MessageImageContainer = styled(Box)({
  width: '100%',
  display: 'flex',
  justifyContent: 'flex-end',
  marginBottom: '8px',
});

const MessageImage = styled('img')({
  maxWidth: '70%',
  maxHeight: '150px',
  borderRadius: '4px',
  boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
});

const DropZone = styled(Box)({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 120, 212, 0.2)',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  zIndex: 10,
  pointerEvents: 'none',
  opacity: 0,
  transition: 'opacity 0.3s',
  '&.active': {
    opacity: 1,
  }
});

const ChatBar = () => {
  const [messages, setMessages] = useState([
    { text: 'Hello! How can I help you today?', sender: 'ai', timestamp: Date.now() - 1000 },
  ]);
  const [input, setInput] = useState('');
  const [contextImage, setContextImage] = useState(null);
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const messagesEndRef = React.useRef(null);
  const chatContainerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  React.useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() || contextImage) {
      const userMessage = { 
        text: input, 
        sender: 'user', 
        timestamp: Date.now(),
        image: contextImage
      };
      setMessages([...messages, userMessage]);
      setInput('');
      
      // Simulate AI response
      setTimeout(() => {
        const aiMessage = {
          text: contextImage 
            ? `I see you've shared an image. ${input ? 'You also said: "' + input + '". ' : ''}How can I help with this?` 
            : 'I understand you sent: "' + input + '". How can I help further?',
          sender: 'ai',
          timestamp: Date.now()
        };
        setMessages(prev => [...prev, aiMessage]);
      }, 1000);
      
      // Clear context image after sending
      setContextImage(null);
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDraggingOver(true);
  };
  
  const handleDragLeave = () => {
    setIsDraggingOver(false);
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDraggingOver(false);
    const imageUrl = e.dataTransfer.getData('text/plain');
    if (imageUrl) {
      setContextImage(imageUrl);
    }
  };

  return (
    <ChatContainer ref={chatContainerRef}>
      <MessagesContainer 
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        sx={{ position: 'relative' }}
      >
        {isDraggingOver && (
          <DropZone className="active">
            <Typography variant="h6" sx={{ color: 'white', backgroundColor: 'rgba(0,0,0,0.7)', padding: '12px 24px', borderRadius: '8px' }}>
              Drop image here to add context
            </Typography>
          </DropZone>
        )}
        {messages.map((message) => (
          <Box
            key={message.timestamp}
            sx={{
              backgroundColor: message.sender === 'user' ? '#0078d4' : '#333',
              color: 'white',
              padding: '8px 12px',
              borderRadius: '8px',
              maxWidth: '80%',
              alignSelf: message.sender === 'user' ? 'flex-end' : 'flex-start',
              margin: '4px 0',
              position: 'relative'
            }}
          >
            {message.image && (
              <MessageImageContainer>
                <MessageImage 
                  src={message.image.startsWith('data:') ? message.image : `data:image/jpeg;base64,${message.image}`} 
                  alt="Shared context" 
                />
              </MessageImageContainer>
            )}
            {message.text}
          </Box>
        ))}
        <div ref={messagesEndRef} />
      </MessagesContainer>
      
      {contextImage && (
        <ContextPreview>
          <ContextImage 
            src={contextImage.startsWith('data:') ? contextImage : `data:image/jpeg;base64,${contextImage}`} 
            alt="Context" 
          />
          <Typography variant="body2" sx={{ color: '#ccc', flexGrow: 1 }}>
            Image added as context
          </Typography>
          <IconButton 
            size="small" 
            onClick={() => setContextImage(null)}
            sx={{ color: '#999' }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </ContextPreview>
      )}
      
      <InputContainer>
        <form onSubmit={handleSubmit}>
          <StyledInput
            placeholder="Ask Xactly"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            variant="outlined"
            size="small"
          />
        </form>
      </InputContainer>
    </ChatContainer>
  );
};

export default ChatBar;
