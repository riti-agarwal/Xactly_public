import React from 'react';
import { Box, Typography, styled } from '@mui/material';
import ProductGrid from './components/ProductGrid';
import ChatBar from './components/ChatBar';
import ProductCollections from './components/ProductCollections';
import { SidebarProvider } from './context/SidebarContext';

const MainContainer = styled(Box)({
  display: 'flex',
  flexDirection: 'column',
  height: '100vh',
  overflow: 'hidden',
  backgroundColor: '#1e1e1e',
});

const ProductSection = styled(Box)({
  flex: '2',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  position: 'relative', // Added for absolute positioning of collections
});

const NavBar = styled(Box)({
  padding: '12px 24px',
  backgroundColor: '#1a1a1a',
  borderBottom: '1px solid #333',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  zIndex: 10,
});

const ProductContent = styled(Box)({
  flex: 1,
  overflowY: 'auto',
  padding: '20px',
});

const ChatSection = styled(Box)({
  flex: '1',
  borderLeft: '1px solid #333',
  display: 'flex',
  flexDirection: 'column',
});

const ContentContainer = styled(Box)({
  display: 'flex',
  flex: 1,
  overflow: 'hidden',
});

function App() {
  return (
    <SidebarProvider>
      <MainContainer>
        <NavBar>
          <Typography variant="h6" sx={{ fontWeight: 600, color: 'white', letterSpacing: '0.5px' }}>
            Xactly
          </Typography>
        </NavBar>
        <ContentContainer>
          <ProductSection>
            <ProductContent>
              <ProductGrid />
            </ProductContent>
            <ProductCollections />
          </ProductSection>
          <ChatSection>
            <ChatBar />
          </ChatSection>
        </ContentContainer>
      </MainContainer>
    </SidebarProvider>
  );
}

export default App;
